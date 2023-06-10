import os
import shutil
import argparse
import pandas as pd
import copy
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
                pbar.update(1)
                if ".jpg" not in file:
                    continue

                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_root, file)
                shutil.copy2(src_file, dst_file)

import re, string, unicodedata
import re
import csv
import pandas as pd

class Preprocessing:
    def __init__(self, train_df=None, test_df=None, val_df=None):
        print("Beginning preprocessing")
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df

    def replace_contractions(self, text):
        """Replace contractions in string of text"""
        return text

    def remove_URL(self, sample):
        """Remove URLs from a sample string"""
        domain_pattern = re.compile(r"\b(?:\w+\.)+\w+\b", re.IGNORECASE)
        sample = re.sub(domain_pattern, "", sample)
        sample = re.sub(r"\S+\.[(net)|(com)|(org)]\S+", "", sample)
        sample = re.sub(r"http\S+", "", sample)
        sample = re.sub(r"\d+", " ", sample)
        sample = re.sub(r"\s+", " ", sample)
        sample = re.sub(r"_", " ", sample)
        return sample

    def remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(self, words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^:\w\s]', ' ', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def normalize(self, words):
        words = self.remove_non_ascii(words)
        words = self.to_lowercase(words)
        # words = self.remove_punctuation(words)
        return words

    def remove_extra_spaces(self, text):
        """Remove extra spaces in a string"""
        return re.sub(r'\s+', ' ', text.strip())
        
    def handle_mami_row(self, sample):
        sample = self.remove_URL(sample)
        # Tokenize
        words = sample.split(' ')
        words = self.normalize(words)

        normalized_text = ''
        for w in words:
            normalized_text += w+' '
        
        normalized_text = self.remove_extra_spaces(normalized_text)

        return normalized_text.strip()

    def handle_df(self, input_df):
        df = input_df.copy()
        preprocessed_texts = []
        for index, row in df.iterrows():
            preprocessed_text = self.handle_mami_row(row['text'])
            preprocessed_texts.append(preprocessed_text)
        df['text'] = preprocessed_texts
        return df

    def preprocess_mami(self):
        if self.train_df is not None:
            ptrain_df = self.handle_df(self.train_df)
            print(ptrain_df["text"].equals(self.train_df["text"]))
        if self.test_df is not None:
            ptest_df = self.handle_df(self.test_df)
        if self.val_df is not None:
            pval_df =self.handle_df(self.val_df)
        
        return ptrain_df, ptest_df, pval_df



def main(dataset_dir: str, processed_dir: str, process_data: str):

    train_fp = os.path.join(dataset_dir, "training", "TRAINING", "training.csv")
    test_fp = os.path.join(dataset_dir, "test", "test", "Test.csv")
    label_fp = os.path.join(dataset_dir, "test_labels.txt")
    val_fp = os.path.join(dataset_dir, "trial", "Users", "fersiniel", "Desktop", "MAMI - TO LABEL", "TRIAL DATASET", "trial.csv")
    
    ## copy images
    image_train_dir = os.path.join(dataset_dir, "training", "TRAINING")
    image_val_dir = os.path.join(dataset_dir, "trial", "Users", "fersiniel", "Desktop", "MAMI - TO LABEL", "TRIAL DATASET")
    image_test_dir = os.path.join(dataset_dir, "test", "test")

    dataset_train_dir = os.path.join(processed_dir, "mami", "images", "train")
    dataset_val_dir = os.path.join(processed_dir, "mami", "images", "validate")
    dataset_test_dir = os.path.join(processed_dir, "mami", "images", "test")
    os.makedirs(dataset_train_dir, exist_ok=True)
    os.makedirs(dataset_val_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # copy_folder_with_progress(image_train_dir, dataset_train_dir)
    # copy_folder_with_progress(image_val_dir, dataset_val_dir)
    # copy_folder_with_progress(image_test_dir, dataset_test_dir)

    # derive misogynous, shaming, stereotype, objectification and violence

    train_df = pd.read_csv(train_fp, sep="\t")
    train_df = train_df.rename({
        "file_name": "img",
        "Text Transcription": "text"
    }, axis=1)

    test_df = pd.read_csv(test_fp, sep="\t")
    test_df = test_df.rename({
        "file_name": "img",
        "Text Transcription": "text"
    }, axis=1)

    ## handle the merging of annotations and labels for test
    col_names = ["file_name", "misogynous","shaming","stereotype","objectification","violence"]
    label_df = pd.read_csv(label_fp, sep="\t", names=col_names)

    test_df["misogynous"] = label_df["misogynous"].copy()
    test_df["shaming"] = label_df["shaming"].copy()
    test_df["stereotype"] = label_df["stereotype"].copy()
    test_df["objectification"] = label_df["objectification"].copy()
    test_df["violence"] = label_df["violence"].copy()
    
    val_df = pd.read_csv(val_fp, sep="\t")
    val_df = val_df.rename({
        "file_name": "img",
        "Text Transcription": "text"
    }, axis=1)
   
    # modifying id column
    train_df['id'] = range(0, len(train_df))
    test_df['id'] = range(len(train_df), len(train_df) + len(test_df))
    val_df['id'] = range(len(train_df) + len(test_df), len(train_df) + len(test_df) + len(val_df))

    # handle preprocessing
    if process_data == "True":
        process = Preprocessing(train_df=train_df, test_df=test_df, val_df=val_df)
        train_df, test_df, val_df = process.preprocess_mami()

    # create the new original file
    new_train_fp = os.path.join(processed_dir, "mami", "annotations", "train.jsonl")
    new_val_fp = os.path.join(processed_dir, "mami", "annotations", "validate.jsonl")
    new_test_fp = os.path.join(processed_dir, "mami", "annotations", "test.jsonl")
    os.makedirs(os.path.dirname(new_train_fp), exist_ok=True)

    train_df.to_json(new_train_fp, orient="records", lines=True)
    test_df.to_json(new_test_fp, orient="records", lines=True)
    val_df.to_json(new_val_fp, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converting MAMI dataset to specified format")
    parser.add_argument("--dataset-dir", help="Folder path to the MAMI directory")
    parser.add_argument("--processed-dir", help="Folder path to store the processed MAMI dataset")
    parser.add_argument("--process-data", default="False", help="Enter True or False to preprocess the MAMI dataset")
    args = parser.parse_args()

    main(
        args.dataset_dir,
        args.processed_dir,
        args.process_data
    )