import os
import argparse
import pandas as pd
import shutil
from tqdm import tqdm

def copy_folder_with_progress(src_folder, dst_folder):
    total_files = sum([len(files) for _, _, files in os.walk(src_folder)])
    
    with tqdm(total=total_files, unit='file') as pbar:
        for root, dirs, files in os.walk(src_folder):
            # Create corresponding directories in the destination folder
            dst_root = os.path.join(dst_folder, os.path.relpath(root, src_folder))
            os.makedirs(dst_root, exist_ok=True)

            # Copy files to the destination folder
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_root, file)
                shutil.copy2(src_file, dst_file)
                pbar.update(1)

def main(dataset_dir: str, processed_dir: str):

    img_dir = img_dir = os.path.join(dataset_dir, "img")
    img_out_dir = os.path.join(processed_dir, "fhm", "images")
    os.makedirs(img_out_dir, exist_ok=True)

    copy_folder_with_progress(img_dir, img_out_dir)
    
    train_fp = os.path.join(dataset_dir, "train.jsonl")
    dev_seen_fp = os.path.join(dataset_dir, "dev_seen.jsonl")
    dev_unseen_fp = os.path.join(dataset_dir, "dev_unseen.jsonl")
    test_fp = os.path.join(dataset_dir, "test_unseen.jsonl")

    train_df = pd.read_json(train_fp, lines=True, convert_axes=False)
    dev_seen_df = pd.read_json(dev_seen_fp, lines=True, convert_axes=False)
    dev_unseen_df = pd.read_json(dev_unseen_fp, lines=True, convert_axes=False)
    test_df = pd.read_json(test_fp, lines=True, convert_axes=False)

    train_df['img'] = train_df['img'].apply(lambda x: os.path.basename(x))
    dev_seen_df['img'] = dev_seen_df['img'].apply(lambda x: os.path.basename(x))
    dev_unseen_df['img'] = dev_unseen_df['img'].apply(lambda x: os.path.basename(x))
    test_df['img'] = test_df['img'].apply(lambda x: os.path.basename(x))

    train_df['id'] = range(0, len(train_df))
    test_df['id'] = range(len(train_df), len(train_df) + len(test_df))
    dev_unseen_df['id'] = range(len(train_df) + len(test_df), len(train_df) + len(test_df) + len(dev_unseen_df))
    dev_seen_df['id'] = range(len(train_df) + len(test_df) + len(dev_unseen_df), len(train_df) + len(test_df) + len(dev_seen_df)+len(dev_unseen_df))

    # create the new original file
    new_train_fp = os.path.join(processed_dir, "fhm", "annotations", "train.jsonl")
    new_dev_seen_fp = os.path.join(processed_dir, "fhm", "annotations", "dev_seen.jsonl")
    new_dev_unseen_fp = os.path.join(processed_dir, "fhm", "annotations", "dev_unseen.jsonl")
    new_test_fp = os.path.join(processed_dir, "fhm", "annotations", "test.jsonl")
    os.makedirs(os.path.dirname(new_train_fp), exist_ok=True)

    train_df.to_json(new_train_fp, orient="records", lines=True)
    dev_seen_df.to_json(new_dev_seen_fp, orient="records", lines=True)
    dev_unseen_df.to_json(new_dev_unseen_fp, orient="records", lines=True)
    test_df.to_json(new_test_fp, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting Facebook's Hateful Memes dataset to specified format")
    parser.add_argument("--dataset-dir", help="Folder path to the Facebook's Hateful Memes directory")
    parser.add_argument("--processed-dir", help="Folder path to store the processed FHM dataset")
    args = parser.parse_args()

    main(
        args.dataset_dir,
        args.processed_dir
    )