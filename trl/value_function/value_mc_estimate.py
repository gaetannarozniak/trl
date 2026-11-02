import json

import datasets
from vllm import LLM, RequestOutput, SamplingParams

from trl.value_function.reward_function import reward_function


class ValueMC:
    def __init__(self, prompts: list[str], ground_truths: list[str], args):
        self.eval_interval: int = args.eval_interval
        self.n_samples_mc = args.n_samples_mc
        self.max_model_len = args.max_model_len
        self.prompts = prompts
        self.ground_truths = ground_truths
        self.vllm_policy = LLM(model=args.model_id, max_model_len=self.max_model_len)

    def generate_trajs_to_eval(self) -> list[RequestOutput]:
        return self.vllm_policy.generate(
            prompts=self.prompts, sampling_params=SamplingParams(max_tokens=self.max_model_len)
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
        mc_outputs = self.vllm_policy.generate(
            prompts=prefixes, sampling_params=SamplingParams(n=self.n_samples_mc, max_tokens=self.max_model_len)
        )
        tokenizer = self.vllm_policy.get_tokenizer()

        rewards = [[] for _ in range(len(self.prompts))]
        for i, mc_output in enumerate(mc_outputs):
            rewards[traj_mapping[i][0]].append([])
            mean_reward = 0.0
            ground_truth = self.ground_truths[traj_mapping[i][0]]
            prefix_text = tokenizer.decode(prefixes[i])
            for n in range(self.n_samples_mc):
                mean_reward += (
                    reward_function(prefix_text + mc_output.outputs[n].text, ground_truth) / self.n_samples_mc
                )
            rewards[traj_mapping[i][0]][traj_mapping[i][1]] = mean_reward

        return rewards

    def dump(self, evaluated_trajs, values, file_path):
        with open(file_path, "w") as f:
            for prompt, gt, traj, value in zip(self.prompts, self.ground_truths, evaluated_trajs, values, strict=True):
                f.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "ground_truth": gt,
                            "completion_token_ids": traj.outputs[0].token_ids,
                            "completion_string": traj.outputs[0].text,
                            "values": value,
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
