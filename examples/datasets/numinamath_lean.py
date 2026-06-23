from datasets import load_dataset
import argparse

"""
python numinamath_lean.py --output_dir /checkpoint/scientific-reasoning/gaetan/trl/datasets/numinamath_lean
"""

PROMPT = (
    "Prove the following theorem in Lean 4. Think step by step, then output "
    "the complete proof inside a ```lean4 ... ``` code block.\n\n"
    "Theorem:\n"
)

SHUFFLE_SEED = 42
TEST_SIZE = 128

def load_and_prepare_dataset():
    dataset = load_dataset("AI-MO/NuminaMath-LEAN", split="train")
    dataset = dataset.filter(lambda x: x["ground_truth_type"] == "complete")
    dataset = dataset.map(
        lambda x: {"prompt": [{"role": "user", "content": PROMPT + x["formal_statement"]}]},
        remove_columns=dataset.column_names,
    )

    dataset = dataset.shuffle(seed=SHUFFLE_SEED)
    test = dataset.select(range(TEST_SIZE))
    train = dataset.select(range(TEST_SIZE, len(dataset)))
    return train, test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    train, test = load_and_prepare_dataset()
    train.to_json(args.output_dir+"/train.jsonl")
    test.to_json(args.output_dir+"/test.jsonl")

if __name__ == "__main__":
    main()
