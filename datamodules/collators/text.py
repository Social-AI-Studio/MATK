import torch
from torch.nn.utils.rnn import pad_sequence

def text_collate_fn(batches, labels):

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
        key = f"{label}_input_ids"
        labels_dict[label] = torch.tensor([i[label] for i in items], dtype=torch.int64)
        labels_dict["labels"].extend([i[key] for i in items])

        # Capture the label indices for each dataset
        indices = list(range(start_index, start_index + len(items)))
        start_index = len(items)
        labels_dict[f"{label}_indices"] = torch.tensor(indices, dtype=torch.int64)

    labels_dict["labels"] = pad_sequence(
        labels_dict["labels"],
        batch_first=True, 
        padding_value=0
    )

    input_ids, attention_mask = [], []
    for item in flattened_batches:
        input_ids.append(item["input_ids"])
        attention_mask.append(item["attention_mask"])
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Get Labels
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    inputs.update(labels_dict)

    return inputs