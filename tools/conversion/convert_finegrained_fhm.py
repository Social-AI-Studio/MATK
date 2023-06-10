import os
import argparse
import pandas as pd
import shutil

# binary classification
HATEFULNESS = {
    v:k for k,v in enumerate([
        "not_hateful",
        "hateful"
    ])
}

# 6-class multi-label classification
PROTECTED_CATEGORY = {
    v:k for k,v in enumerate([
        "pc_empty",
        "disability",
        "nationality",
        "race",
        "religion",
        "sex"
    ])
}

# 8-class multi-label classification
PROTECTED_ATTACK = {
    v:k for k,v in enumerate([
        "attack_empty",
        "contempt",
        "dehumanizing",
        "exclusion",
        "inciting_violence",
        "inferiority",
        "mocking",
        "slurs"
    ])
}

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

    train_df['img'] = train_df['img'].apply(lambda x: os.path.basename(x))
    train_df['hate'] = train_df['gold_hate'].apply(lambda x: HATEFULNESS[x[0]])
    train_df['pc'] = train_df['gold_pc'].apply(lambda x: [PROTECTED_CATEGORY[i] for i in x])
    train_df['attack'] = train_df['gold_attack'].apply(lambda x: [PROTECTED_ATTACK[i] for i in x])
    
    dev_seen_df['img'] = dev_seen_df['img'].apply(lambda x: os.path.basename(x))
    dev_seen_df['hate'] = dev_seen_df['gold_hate'].apply(lambda x: HATEFULNESS[x[0]])
    dev_seen_df['pc'] = dev_seen_df['gold_pc'].apply(lambda x: [PROTECTED_CATEGORY[i] for i in x])
    dev_seen_df['attack'] = dev_seen_df['gold_attack'].apply(lambda x: [PROTECTED_ATTACK[i] for i in x])

    dev_unseen_df['img'] = dev_unseen_df['img'].apply(lambda x: os.path.basename(x))
    dev_unseen_df['hate'] = dev_unseen_df['gold_hate'].apply(lambda x: HATEFULNESS[x[0]])
    dev_unseen_df['pc'] = dev_unseen_df['gold_pc'].apply(lambda x: [PROTECTED_CATEGORY[i] for i in x])
    dev_unseen_df['attack'] = dev_unseen_df['gold_attack'].apply(lambda x: [PROTECTED_ATTACK[i] for i in x])

    test_df['img'] = test_df['img'].apply(lambda x: os.path.basename(x))

    # modifying id column
    train_df['id'] = range(0, len(train_df))
    test_df['id'] = range(len(train_df), len(train_df) + len(test_df))
    dev_unseen_df['id'] = range(len(train_df) + len(test_df), len(train_df) + len(test_df) + len(dev_unseen_df))
    dev_seen_df['id'] = range(len(train_df) + len(test_df) + len(dev_unseen_df), len(train_df) + len(test_df) + len(dev_seen_df)+len(dev_unseen_df))

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