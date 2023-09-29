import torch

def text_collate_fn(batch, tokenizer, labels):
    texts = []
    for item in batch:
        texts.append(item["text"])
    
    inputs = tokenizer(  
        text=texts, 
        padding=True,
        return_tensors="pt"
    )

    # Get Labels
    for l in labels:
        labels = [feature[l] for feature in batch]
        labels = tokenizer(labels, padding=True, truncation=True, return_tensors="pt").input_ids

        inputs[l] = labels

    return inputs