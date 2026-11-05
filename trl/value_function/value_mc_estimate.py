import json
import os

import datasets
from vllm import LLM, RequestOutput, SamplingParams

from trl.value_function.reward_function import reward_function


class ValueMC:
    def __init__(self, prompts: list[str], ground_truths: list[str], args):
        self.model_id = args.model_id
        self.eval_interval: int = args.eval_interval
        self.n_samples_mc = args.n_samples_mc
        self.max_model_len = args.max_model_len
        self.max_completion_length = args.max_completion_length
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.vllm_policy = LLM(model=args.model_id, max_model_len=self.max_model_len)

    def generate_trajs_to_eval(self) -> list[RequestOutput]:
        return self.vllm_policy.generate(
            prompts=self.prompts, sampling_params=SamplingParams(max_tokens=self.max_completion_length)
        )

    def generate_prefixes(self, outputs: list[RequestOutput]) -> tuple[list[list[int]], list[tuple[int, int]]]:
        traj_token_ids = [output.prompt_token_ids + output.outputs[0].token_ids for output in outputs]
        prefixes = []

        traj_mapping = []
        for i, traj in enumerate(traj_token_ids):
            prompt_length = len(outputs[i].prompt_token_ids)
            for k, t in enumerate(range(prompt_length, len(traj), self.eval_interval)):
                prefixes.append(traj[:t])
                traj_mapping.append((i, k))
        return prefixes, traj_mapping

    def eval_prefixes(self, prefixes: list[list[int]], traj_mapping: list[tuple[int, int]]) -> list[list[float]]:
        # Prefix k already holds k * eval_interval completion tokens, so cap the total at max_completion_length
        sampling_params = [
            SamplingParams(n=self.n_samples_mc, max_tokens=self.max_completion_length - k * self.eval_interval)
            for _, k in traj_mapping
        ]
        mc_outputs = self.vllm_policy.generate(
            prompts=[{"prompt_token_ids": prefix} for prefix in prefixes], sampling_params=sampling_params
        )
        tokenizer = self.vllm_policy.get_tokenizer()

        rewards = [[] for _ in range(len(self.prompts))]
        for i, mc_output in enumerate(mc_outputs):
            rewards[traj_mapping[i][0]].append([])
            ground_truth = self.ground_truths[traj_mapping[i][0]]
            completion_start = len(prefixes[i]) - traj_mapping[i][1] * self.eval_interval
            prefix_text = tokenizer.decode(prefixes[i][completion_start:], skip_special_tokens=True)
            # Sum then divide, so that always/never solved prefixes are exactly 1.0/0.0
            mean_reward = (
                sum(reward_function(prefix_text + output.text, ground_truth) for output in mc_output.outputs)
                / self.n_samples_mc
            )
            rewards[traj_mapping[i][0]][traj_mapping[i][1]] = mean_reward

        return rewards

    def dump(self, evaluated_trajs, values, file_path):
        sampling_params = SamplingParams()
        mc_args = {
            "model_id": os.path.realpath(self.model_id) if os.path.isdir(self.model_id) else self.model_id,
            "eval_interval": self.eval_interval,
            "max_completion_length": self.max_completion_length,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "top_k": sampling_params.top_k,
            "min_p": sampling_params.min_p,
            "repetition_penalty": sampling_params.repetition_penalty,
        }
        with open(file_path, "w") as f:
            for prompt, gt, traj, value in zip(self.prompts, self.ground_truths, evaluated_trajs, values, strict=True):
                f.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "ground_truth": gt,
                            "prompt_token_ids": traj.prompt_token_ids,
                            "completion_token_ids": traj.outputs[0].token_ids,
                            "completion_string": traj.outputs[0].text,
                            "values": value,
                            "mc_args": mc_args,
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--n_samples_mc", type=int, required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--eval_interval", type=int, required=True)
    parser.add_argument("--max_model_len", type=int, required=True)
    parser.add_argument("--max_completion_length", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_prompts", type=int, default=None)

    args = parser.parse_args()

    dataset = datasets.load_dataset(args.dataset_name, split="train").shuffle(seed=0)
    if args.max_prompts is not None:
        dataset = dataset.select(range(args.max_prompts))
    dataset = dataset.shard(num_shards=args.shard_count, index=args.shard_index, contiguous=False)
    prompts = [row[0]["content"] for row in dataset["prompt"]]
    ground_truths = dataset["label"]
    value_mc = ValueMC(prompts, ground_truths, args)
    outputs = value_mc.generate_trajs_to_eval()
    prefixes, traj_mapping = value_mc.generate_prefixes(outputs)
    rewards = value_mc.eval_prefixes(prefixes, traj_mapping)
    value_mc.dump(outputs, rewards, args.out)
    value_mc.vllm_policy.llm_engine.engine_core.shutdown()
    print("Dumped with success!")
