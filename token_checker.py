import os
import argparse
from datasets.utils import _load_jsonl
from transformers import AutoTokenizer
import tqdm
import pickle as pkl
import statistics

def load_text(dataset_dir: str):
    data = _load_jsonl(dataset_dir)
    batch = []
    # translate labels into numeric values
    for record in tqdm.tqdm(data, desc="Preprocessing labels"):
        input = {}
        record["img"] = os.path.basename(record["img"])
        batch.append(record)
    return batch

def load_captions(caption_dir):
    data = {}
    
    with open(caption_dir, "rb") as f:
        data = pkl.load(f)

    return data

def tokenize(batch, tokenizer):
    for item in batch:
        
        concat = item["text"] + item["caption"]
        item["concat"] = concat

        item["text_tokens"] = tokenizer(item["text"], padding=True,
        return_tensors="pt")
        item["text_tokens_length"] = len(item["text_tokens"]["input_ids"][0])

        item["caption_tokens"] = tokenizer(item["caption"], padding=True,
        return_tensors="pt")
        item["caption_tokens_length"] = len(item["caption_tokens"]["input_ids"][0])

        item["concat_tokens"] = tokenizer(concat, padding=True,
        return_tensors="pt")
        item["concat_tokens_length"] = len(item["concat_tokens"]["input_ids"][0])

    return batch

def combine_text_and_caption_dicts(text_data, caption_data):

    combined = text_data
    for record in text_data:
        record["caption"] = caption_data[record["id"]]

    return combined

def analyze_data(data_list, tokenizer):
    max_text_tokens_length = 0
    max_caption_tokens_length = 0
    max_concat_tokens_length = 0
    stddev_text_tokens_length = 0
    stddev_caption_tokens_length = 0
    stddev_concat_tokens_length = 0
    truncated_examples = []

    text_tokens_lengths = []
    caption_tokens_lengths = []
    concat_tokens_lengths = []

    for item in data_list:
        text_len = item["text_tokens_length"]
        caption_len = item["caption_tokens_length"]
        concat_len = item["concat_tokens_length"]

        max_text_tokens_length = max(max_text_tokens_length, text_len)
        max_caption_tokens_length = max(max_caption_tokens_length, caption_len)
        max_concat_tokens_length = max(max_concat_tokens_length, concat_len)

        text_tokens_lengths.append(text_len)
        caption_tokens_lengths.append(caption_len)
        concat_tokens_lengths.append(concat_len)

        if concat_len > tokenizer.model_max_length:
            truncated_examples.append(item)

    stddev_text_tokens_length = statistics.stdev(text_tokens_lengths)
    stddev_caption_tokens_length = statistics.stdev(caption_tokens_lengths)
    stddev_concat_tokens_length = statistics.stdev(concat_tokens_lengths)

    return {
        "max_text_tokens_length": max_text_tokens_length,
        "max_caption_tokens_length": max_caption_tokens_length,
        "max_concat_tokens_length": max_concat_tokens_length,
        "stddev_text_tokens_length": stddev_text_tokens_length,
        "stddev_caption_tokens_length": stddev_caption_tokens_length,
        "stddev_concat_tokens_length": stddev_concat_tokens_length,
        "truncated_examples": len(truncated_examples)
    }

def main(dataset_dir: str, model_dir: str, caption_dir):

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    data = load_text(dataset_dir)
    captions = load_captions(caption_dir)
    combined_data = combine_text_and_caption_dicts(data, captions)
    tokenized_data = tokenize(combined_data, tokenizer)

    stats = analyze_data(tokenized_data, tokenizer)
    print(stats)
    
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Making sure the number of tokens doesn't exceed the model's token limit.")
    parser.add_argument("--dataset-dir", help="Folder path to the Facebook's Hateful Memes directory")
    parser.add_argument("--model-dir", help="Folder path tokenizer/model")
    parser.add_argument("--caption-dir", help="Folder path captions")
    args = parser.parse_args()

    main(
        args.dataset_dir,
        args.model_dir,
        args.caption_dir
    )
