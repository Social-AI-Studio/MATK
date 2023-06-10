import torch

def text_collate_fn(batch, tokenizer, labels):
    texts = []
    for item in batch:
        texts.append(item["text"])
    
    inputs = tokenizer(  
        text=texts, return_tensors="pt", padding=True
    )

    # Get Labels
    for l in labels:
        if l in batch[0].keys():
            labels = [feature[l] for feature in batch]

            if isinstance(labels[0], str):
                labels = tokenizer(labels, truncation=True, return_tensors="pt", add_special_tokens=False).input_ids
                labels[labels == tokenizer.pad_token_id] = -100
            else:
                labels = torch.tensor(labels, dtype=torch.int64)

            inputs[l] = labels

    return inputs