import torch
import numpy as np

def _common_collate(
        batch, 
        tokenizer,
        labels
    ):

    texts = []
    for item in batch:
        texts.append(item["text"])
    
    inputs = tokenizer(
            texts,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

    # Get Labels
    for l in labels:
        if l in batch[0].keys():
            labels = [feature[l] for feature in batch]
            inputs[l] = torch.tensor(labels, dtype=torch.int64)
            
    return inputs

def frcnn_collate_fn(
        batch, 
        tokenizer,
        image_preprocess,
        labels
    ):
    
    inputs = _common_collate(
        batch=batch,
        tokenizer=tokenizer,
        labels=labels
    )

    images = [item['image_path'] for item in batch]
    images, sizes, scales_yx = image_preprocess(images)

    inputs["images"] = images
    inputs["sizes"] = sizes
    inputs["scales_yx"] = scales_yx
    
    return inputs

def frcnn_collate_fn_fast(
        batch, 
        tokenizer,
        labels
    ):
    inputs = _common_collate(
        batch=batch,
        tokenizer=tokenizer,
        labels=labels
    )


    visual_feats, visual_pos = [], []
    for item in batch:
        visual_feats.append(item["roi_features"])
        visual_pos.append(item["normalized_boxes"])

    inputs['visual_feats'] = torch.cat(visual_feats, dim=0)
    inputs['visual_pos'] = torch.cat(visual_pos, dim=0)

    return inputs