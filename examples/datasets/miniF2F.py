from datasets import load_dataset
import argparse

"""
python miniF2F.py --output_dir /checkpoint/scientific-reasoning/gaetan/trl/datasets/miniF2F
"""

PROMPT = (
    "Prove the following theorem in Lean 4. Think step by step, then output "
    "the complete proof inside a ```lean4 ... ``` code block.\n\n"
    "Theorem:\n"
)

def load_and_prepare_dataset():
    # Login using e.g. `huggingface-cli login` to access this dataset
    dataset = load_dataset("Tonic/MiniF2F", split="train")

    dataset = dataset.map(lambda x: {"prompt": PROMPT + x["formal_statement"]})
    dataset = dataset.remove_columns(
      ["informal_prefix", "goal", "header", "name", "formal_statement"]
    )

    valid = dataset.filter(lambda x: x["split"] == "valid").remove_columns("split")
    test = dataset.filter(lambda x: x["split"] == "test").remove_columns("split")
    test32 = test.shuffle().select(range(32))
    return valid, test, test32

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    valid, test, test32 = load_and_prepare_dataset()
    valid.to_json(args.output_dir+"/valid.jsonl")
    test.to_json(args.output_dir+"/test.jsonl")
    test32.to_json(args.output_dir+"/test32.jsonl")

if __name__ == "__main__":
    main()
