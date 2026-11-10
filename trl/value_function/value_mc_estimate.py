import json
import os
import time

import datasets
from vllm import LLM, SamplingParams

from trl.value_function.reward_function import reward_function


# Problems whose prompt value, estimated on the screening rollouts, is outside this range are dropped
ELIGIBLE_VALUES = (0.1, 0.9)
# The MC rollouts are generated in this many vLLM calls, each followed by a progress line. The batch drains at the end
# of each call, so fewer calls are slightly faster
PROGRESS_STEPS = 20


class ValueMC:
    def __init__(self, indices: list[int], prompts: list[str], ground_truths: list[str], args):
        self.model_id = args.model_id
        self.eval_interval: int = args.eval_interval
        self.n_samples_mc = args.n_samples_mc
        self.num_trajectories = args.num_trajectories
        self.screen_samples = args.screen_samples
        self.max_model_len = args.max_model_len
        self.max_completion_length = args.max_completion_length
        self.shard_index = args.shard_index
        self.indices = indices
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.vllm_policy = LLM(model=args.model_id, max_model_len=self.max_model_len)

    def generate_trajs_to_eval(self) -> list[dict]:
        # The first `screen_samples` rollouts of each prompt estimate its value, which selects the problems, and the
        # last `num_trajectories`, independent of this selection, are the trajectories to evaluate
        start = time.perf_counter()
        sampling_params = SamplingParams(
            n=self.screen_samples + self.num_trajectories, max_tokens=self.max_completion_length
        )
        outputs = self.vllm_policy.generate(prompts=self.prompts, sampling_params=sampling_params, use_tqdm=False)
        trajs = []
        for index, prompt, ground_truth, output in zip(
            self.indices, self.prompts, self.ground_truths, outputs, strict=True
        ):
            screen_rewards = [
                reward_function(sample.text, ground_truth) for sample in output.outputs[: self.screen_samples]
            ]
            screen_value = sum(screen_rewards) / self.screen_samples
            if not ELIGIBLE_VALUES[0] <= screen_value <= ELIGIBLE_VALUES[1]:
                continue
            for trajectory, sample in enumerate(output.outputs[self.screen_samples :]):
                trajs.append(
                    {
                        "index": index,
                        "trajectory": trajectory,
                        "prompt": prompt,
                        "ground_truth": ground_truth,
                        "screen_value": screen_value,
                        "prompt_token_ids": output.prompt_token_ids,
                        "completion_token_ids": list(sample.token_ids),
                        "completion_string": sample.text,
                        "reward": reward_function(sample.text, ground_truth),
                    }
                )
        print(
            f"shard {self.shard_index}: screened {len(self.prompts)} problems in "
            f"{(time.perf_counter() - start) / 60:.1f} min, {len(trajs) // self.num_trajectories} eligible",
            flush=True,
        )
        return trajs

    def eval_prefixes(self, trajs: list[dict]) -> None:
        # Adds to each trajectory the number of successful rollouts out of `n_samples_mc` after every `eval_interval`
        # completion tokens, the first one after the prompt alone
        tokenizer = self.vllm_policy.get_tokenizer()
        start, num_tokens = time.perf_counter(), 0
        batch_size = max(1, -(-len(trajs) // PROGRESS_STEPS))
        for batch_start in range(0, len(trajs), batch_size):
            batch = trajs[batch_start : batch_start + batch_size]
            prefixes, sampling_params, owners = [], [], []
            for traj in batch:
                traj["successes"] = []
                for offset in range(0, len(traj["completion_token_ids"]), self.eval_interval):
                    prefixes.append(
                        {"prompt_token_ids": traj["prompt_token_ids"] + traj["completion_token_ids"][:offset]}
                    )
                    # The prefix already holds `offset` completion tokens, so cap the total at max_completion_length
                    sampling_params.append(
                        SamplingParams(n=self.n_samples_mc, max_tokens=self.max_completion_length - offset)
                    )
                    owners.append((traj, offset))
            outputs = self.vllm_policy.generate(prompts=prefixes, sampling_params=sampling_params, use_tqdm=False)
            for (traj, offset), output in zip(owners, outputs, strict=True):
                prefix_text = tokenizer.decode(traj["completion_token_ids"][:offset], skip_special_tokens=True)
                rewards = [
                    reward_function(prefix_text + sample.text, traj["ground_truth"]) for sample in output.outputs
                ]
                traj["successes"].append(int(sum(rewards)))
                num_tokens += sum(len(sample.token_ids) for sample in output.outputs)
            done, elapsed = batch_start + len(batch), time.perf_counter() - start
            print(
                f"shard {self.shard_index}: MC rollouts of {done}/{len(trajs)} trajectories, {num_tokens:.2e} tokens "
                f"in {elapsed / 60:.1f} min, ETA {elapsed / done * (len(trajs) - done) / 60:.0f} min",
                flush=True,
            )

    def dump(self, trajs, file_path):
        sampling_params = SamplingParams()
        mc_args = {
            "model_id": os.path.realpath(self.model_id) if os.path.isdir(self.model_id) else self.model_id,
            "eval_interval": self.eval_interval,
            "n_samples_mc": self.n_samples_mc,
            "num_trajectories": self.num_trajectories,
            "screen_samples": self.screen_samples,
            "max_completion_length": self.max_completion_length,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "top_k": sampling_params.top_k,
            "min_p": sampling_params.min_p,
            "repetition_penalty": sampling_params.repetition_penalty,
        }
        # Renamed once complete, so that run_value_mc.sbatch never merges a partially written shard
        with open(file_path + ".tmp", "w") as f:
            for traj in trajs:
                values = [successes / self.n_samples_mc for successes in traj["successes"]]
                f.write(json.dumps({**traj, "values": values, "mc_args": mc_args}) + "\n")
        os.replace(file_path + ".tmp", file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--n_samples_mc", type=int, required=True)
    parser.add_argument("--num_trajectories", type=int, default=4, help="Trajectories evaluated per problem.")
    parser.add_argument("--screen_samples", type=int, default=64, help="Rollouts estimating the value of a prompt.")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--eval_interval", type=int, required=True)
    parser.add_argument("--max_model_len", type=int, required=True)
    parser.add_argument("--max_completion_length", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_prompts", type=int, default=None)

    args = parser.parse_args()

    dataset = datasets.load_dataset(args.dataset_name, split="train").shuffle(seed=0)
    # Indices in the shuffled dataset, which train_value.py splits into held-out and training problems
    indices = list(range(args.max_prompts or len(dataset)))[args.shard_index :: args.shard_count]
    dataset = dataset.select(indices)
    prompts = [row[0]["content"] for row in dataset["prompt"]]
    ground_truths = dataset["label"]
    value_mc = ValueMC(indices, prompts, ground_truths, args)
    trajs = value_mc.generate_trajs_to_eval()
    value_mc.eval_prefixes(trajs)
    value_mc.dump(trajs, args.out)
    value_mc.vllm_policy.llm_engine.engine_core.shutdown()
    print("Dumped with success!")
