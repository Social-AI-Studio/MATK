import torch

def image_collate_fn(batch, processor):
    texts, images = [], []
    for item in batch:
        texts.append(item["text"])
        images.append(item["image"])
    
    inputs = processor(  
        text=texts, images=images, return_tensors="pt", padding=True
    )

    # Get Labels
    label_name = "label"
    if label_name in batch[0].keys():
        labels = [feature[label_name] for feature in batch]
        inputs['labels'] = torch.tensor(labels, dtype=torch.int64)

    return inputs

def text_collate_fn(batch, tokenizer):
    texts = []
    for item in batch:
        texts.append(item["text"])
    
    inputs = tokenizer(  
        text=texts, return_tensors="pt", padding=True
    )

    # Get Labels
    label_name = "label"
    if label_name in batch[0].keys():
        labels = [feature[label_name] for feature in batch]

        if isinstance(labels[0], str):
            labels = tokenizer(labels, truncation=True, return_tensors="pt", add_special_tokens=False).input_ids
            labels[labels == tokenizer.pad_token_id] = -100
        else:
            labels = torch.tensor(labels, dtype=torch.int64)
        
        inputs['labels'] = labels

    return inputs