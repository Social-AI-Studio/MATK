import os
import torch

def intmeme_collate_fn(batches, processor, tokenizer, labels):    

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


    texts, images, passages = [], [], []
    for item in flattened_batches:
        texts.append(item["text"])
        images.append(item["img"])
        passages.append(item["passage"])
    
    multimodal_inputs = processor(  
        text=texts, images=images, return_tensors="pt", padding=True, truncation=True
    )

    passage_inputs = tokenizer(  
        text=passages, return_tensors="pt", padding=True, truncation=True
    )

    inputs = {
        "meme_input_ids": multimodal_inputs.input_ids,
        "meme_attention_mask": multimodal_inputs.attention_mask,
        "pixel_values": multimodal_inputs.pixel_values,
        "passage_input_ids": passage_inputs.input_ids,
        "passage_attention_mask": passage_inputs.attention_mask,
    }
    inputs.update(labels_dict)

    return inputs