import torch

def text_collate_fn(batches, tokenizer, labels):

    # Restructure the dataset batches
    batch_dict = {label: [] for label in labels}
    for item_tuple in batches:
        for item in item_tuple:
            for label in labels:
                if label in item:
                    batch_dict[label].append(item)
                    continue

    flattened_batches = []
    labels_dict = {
        "labels": []
    }
    start_index = 0
    for label, items in batch_dict.items():
        # Flatten the batches in the order of datasets: fhm -> mami -> ...
        flattened_batches.extend(items)

        # Capture the labels for each dataset
        labels_dict[label] = tokenizer(
            [i[label] for i in items], return_tensors="pt"
        ).input_ids
        labels_dict["labels"].extend([i[label] for i in items])

        # Capture the label indices for each dataset
        indices = list(range(start_index, start_index + len(items)))
        start_index = len(items)
        labels_dict[f"{label}_indices"] = torch.tensor(indices, dtype=torch.int64)

    labels_dict["labels"] = tokenizer(labels_dict["labels"], padding=True, truncation=True, return_tensors="pt").input_ids

    texts = []
    for item in flattened_batches:
        texts.append(item["text"])
    
    inputs = tokenizer(  
        text=texts, 
        padding=True,
        return_tensors="pt"
    )

    # Get Labels
    inputs.update(labels_dict)

    return inputs