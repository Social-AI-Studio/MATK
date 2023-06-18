import os
import argparse
import pandas as pd

def main(dataset_dir: str, processed_dir: str):

    # remap intensity and target
    train_fp = os.path.join(dataset_dir, "data", "annotations", "train.json")
    dev_seen_fp = os.path.join(dataset_dir, "data", "annotations", "dev_seen.json")
    dev_unseen_fp = os.path.join(dataset_dir, "data", "annotations", "dev_unseen.json")
    test_fp = os.path.join(dataset_dir, "data", "annotations", "test.jsonl")

    train_df = pd.read_json(train_fp, lines=True)
    dev_seen_df = pd.read_json(dev_seen_fp, lines=True)
    dev_unseen_df = pd.read_json(dev_unseen_fp, lines=True)
    test_df = pd.read_json(test_fp, lines=True)

    # create the new original file
    new_train_fp = os.path.join(processed_dir, "fhm_finegrained", "annotations", "train.jsonl")
    new_dev_seen_fp = os.path.join(processed_dir, "fhm_finegrained", "annotations", "dev_seen.jsonl")
    new_dev_unseen_fp = os.path.join(processed_dir, "fhm_finegrained", "annotations", "dev_unseen.jsonl")
    new_test_fp = os.path.join(processed_dir, "fhm_finegrained", "annotations", "test.jsonl")
    os.makedirs(os.path.dirname(new_train_fp), exist_ok=True)

    train_df.to_json(new_train_fp, orient="records", lines=True)
    dev_seen_df.to_json(new_dev_seen_fp, orient="records", lines=True)
    dev_unseen_df.to_json(new_dev_unseen_fp, orient="records", lines=True)
    test_df.to_json(new_test_fp, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting Facebook's Hateful Memes finegrained dataset to specified format")
    parser.add_argument("--dataset-dir", help="Folder path to the Facebook's Hateful Memes Fine-Grain directory")
    parser.add_argument("--processed-dir", help="Folder path to store the processed Facebook's Hateful Memes finegrained dataset")
    args = parser.parse_args()

    main(
        args.dataset_dir,
        args.processed_dir
    )