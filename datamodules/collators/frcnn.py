import torch
import numpy as np

def _common_collate(
        batches, 
        tokenizer,
        labels
    ):

    texts = []
    for batch in batches:
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

    indices_list = [item + "_indices" for item in labels]
    
    label_index = 0 
    for l, label_indices_list_name in zip(labels, indices_list):
        label_values = [] 
        label_indices = []
        for batch in batches:
            for dataset_item in batch:
                if l in dataset_item:
                    label_values.append(dataset_item[l])
                    label_indices.append(label_index)
                    label_index+=1
        inputs[l] = torch.tensor(label_values, dtype=torch.int64)
        inputs[label_indices_list_name] = torch.tensor(label_indices, dtype=torch.int64)
            
    return inputs

def frcnn_collate_fn(
        batches, 
        tokenizer,
        image_preprocess,
        labels
    ):
    
    inputs = _common_collate(
        batches=batches,
        tokenizer=tokenizer,
        labels=labels
    )

    images = []
    for batch in batches:
        for item in batch:
            images.append(item["image_path"])
    images, sizes, scales_yx = image_preprocess(images)

    inputs["images"] = images
    inputs["sizes"] = sizes
    inputs["scales_yx"] = scales_yx
    
    return inputs

def frcnn_collate_fn_fast(
        batches, 
        tokenizer,
        labels
    ):
    inputs = _common_collate(
        batches=batches,
        tokenizer=tokenizer,
        labels=labels
    )


    visual_feats, visual_pos = [], []
    for batch in batches:
            for item in batch:
                visual_feats.append(item["roi_features"])
                visual_pos.append(item["normalized_boxes"])

    inputs['visual_feats'] = torch.cat(visual_feats, dim=0)
    inputs['visual_pos'] = torch.cat(visual_pos, dim=0)

    return inputs