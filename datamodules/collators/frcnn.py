import torch
import numpy as np
from typing import Optional, Callable, List

def _common_collate(
    batch,
    tokenizer
):

    texts = []
    for item in batch:
        texts.append(item["templated_text"])

    inputs = tokenizer(
        texts,
        padding="max_length",
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    
    return inputs


def frcnn_collate_fn(
    batches: List[dict],
    tokenizer: Callable,
    labels: List[str],
    image_preprocess: Optional[Callable]
):

    # Restructure the dataset batches
    batch_dict = {label: [] for label in labels}
    for item_tuple in batches:
        for item in item_tuple:
            for label in labels:
                if label in item:
                    batch_dict[label].append(item)
                    continue

    flattened_batches = []
    labels_dict = {}
    start_index = 0
    for label, items in batch_dict.items():
        # Flatten the batches in the order of datasets: fhm -> mami -> ...
        flattened_batches.extend(items)

        # Capture the labels for each dataset
        labels_dict[label] = torch.tensor([i[label] for i in items], dtype=torch.int64)

        # Capture the label indices for each dataset
        indices = list(range(start_index, start_index + len(items)))
        start_index = len(items)
        labels_dict[f"{label}_indices"] = torch.tensor(indices, dtype=torch.int64)

    inputs = _common_collate(
        batch=flattened_batches,
        tokenizer=tokenizer
    )

    if image_preprocess:
        inputs.update(_image_collate_fn(flattened_batches, image_preprocess))
    else:
        inputs.update(_features_collate_fn(flattened_batches))

    inputs.update(labels_dict)

    return inputs


def _features_collate_fn(
    batch
):

    visual_feats, visual_pos = [], []
    for item in batch:
        visual_feats.append(item["roi_features"])
        visual_pos.append(item["normalized_boxes"])

    return {
        'visual_feats': torch.cat(visual_feats, dim=0),
        'visual_pos': torch.cat(visual_pos, dim=0)
    }

def _image_collate_fn(
    batch,
    image_preprocess
):
    images = []
    for item in batch:
        images.append(item["image_path"])

    images, sizes, scales_yx = image_preprocess(images)

    return {
        "images": images,
        "sizes": sizes,
        "scales_yx": scales_yx
    }