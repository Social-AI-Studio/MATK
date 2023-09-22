import torch

def processor_collate_fn(batches, processor, labels):    
    texts, images = [], []

    for batch in batches:
        for item in batch:
            texts.append(item["text"])
            images.append(item["image"])
    
    inputs = processor(  
        text=texts, images=images, return_tensors="pt", padding=True, truncation=True
    )

    # Get Labels
    for l in labels:
        if l in batch[0].keys():
            labels = [feature[l] for feature in batch]
            inputs[l] = torch.tensor(labels, dtype=torch.int64)

    return inputs