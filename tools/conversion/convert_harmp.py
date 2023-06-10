import os
import shutil
import argparse
import pandas as pd

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

INTENSITY_MAP = {
    'not harmful': 0, 
    'somewhat harmful': 1, 
    'very harmful': 2
}

TARGET_MAP = {
    'individual': 0, 
    'organization': 1, 
    'community': 2 , 
    'society': 3
}

def main(dataset_dir: str, processed_dir: str):
    
    train_fp = os.path.join(dataset_dir, "data", "datasets", "memes", "defaults", "annotations", "train.jsonl")
    test_fp = os.path.join(dataset_dir, "data", "datasets", "memes", "defaults", "annotations", "test.jsonl")
    val_fp = os.path.join(dataset_dir, "data", "datasets", "memes", "defaults", "annotations", "val.jsonl")
    
    ## copy images
    img_dir = os.path.join(dataset_dir, "data", "datasets", "memes", "defaults", "images")
    img_out_dir = os.path.join(processed_dir, "harmp", "images")
    os.makedirs(img_out_dir, exist_ok=True)

    copy_folder_with_progress(img_dir, img_out_dir)

    train_df = pd.read_json(train_fp, lines=True)
    val_df = pd.read_json(val_fp, lines=True)
    test_df = pd.read_json(test_fp, lines=True)

    train_df['intensity'] = train_df['labels'].apply(lambda x: INTENSITY_MAP[x[0]])
    train_df['target'] = train_df['labels'].apply(lambda x: TARGET_MAP[x[1]] if len(x) > 1 else -1)
    train_df = train_df.rename({"image": "img"}, axis=1)

    val_df['intensity'] = val_df['labels'].apply(lambda x: INTENSITY_MAP[x[0]])
    val_df['target'] = val_df['labels'].apply(lambda x: TARGET_MAP[x[1]] if len(x) > 1 else -1)
    val_df = val_df.rename({"image": "img"}, axis=1)
    
    test_df['intensity'] = test_df['labels'].apply(lambda x: INTENSITY_MAP[x[0]])
    test_df['target'] = test_df['labels'].apply(lambda x: TARGET_MAP[x[1]] if len(x) > 1 else -1)
    test_df = test_df.rename({"image": "img"}, axis=1)

    # modifying id column
    train_df['id'] = range(0, len(train_df))
    test_df['id'] = range(len(train_df), len(train_df) + len(test_df))
    val_df['id'] = range(len(train_df) + len(test_df), len(train_df) + len(test_df) + len(val_df))

    # create the new original file
    new_train_fp = os.path.join(processed_dir, "harmp", "annotations", "train.jsonl")
    new_val_fp = os.path.join(processed_dir, "harmp", "annotations", "validate.jsonl")
    new_test_fp = os.path.join(processed_dir, "harmp", "annotations", "test.jsonl")
    os.makedirs(os.path.dirname(new_train_fp), exist_ok=True)

    train_df.to_json(new_train_fp, orient="records", lines=True)
    val_df.to_json(new_val_fp, orient="records", lines=True)
    test_df.to_json(new_test_fp, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting Harm-P dataset to specified format")
    parser.add_argument("--dataset-dir", help="Folder path to the Harm-P directory")
    parser.add_argument("--processed-dir", help="Folder path to store the processed dataset")
    args = parser.parse_args()

    main(
        args.dataset_dir,
        args.processed_dir
    )