import torch
from torch.nn.utils.rnn import pad_sequence

def text_collate_fn(tokenizer, batches, labels):

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
        labels_dict["labels"].extend(
            [i[f"templated_{label}"] for i in items]
        )

        # Capture the label indices for each dataset
        indices = list(range(start_index, start_index + len(items)))
        start_index = len(items)
        labels_dict[label] = torch.tensor([i[label] for i in items], dtype=torch.int64)
        labels_dict[f"{label}_indices"] = torch.tensor(indices, dtype=torch.int64)

    tokenized = tokenizer(labels_dict["labels"], padding=True, truncation=True, return_tensors="pt")
    labels_dict["labels_input_ids"] = tokenized.input_ids
    labels_dict["labels_attention_mask"] = tokenized.attention_mask
    del labels_dict["labels"]

    texts = []
    for item in flattened_batches:
        texts.append(item["templated_text"])
    
    inputs = tokenizer(  
        text=texts, 
        padding=True,
        return_tensors="pt"
    )
    inputs.update(labels_dict)

    return inputs

def text_gen_collate_fn(tokenizer, batches):
    # assume no multi task learning for now
    flattened_batches = [item[0] for item in batches]
    labels_dict = {}
    start_index = 0
    indices = list(range(start_index, start_index + len(flattened_batches)))
    labels_dict["indices"] = torch.tensor(indices, dtype=torch.int64)

    texts = []
    targets = []
    for item in flattened_batches:
        texts.append(item["templated_text"])
        targets.append(item["targets"])
    
    inputs = tokenizer(  
        text=texts, 
        padding=True,
        return_tensors="pt"
    )
    targets = tokenizer(  
        text=texts, 
        padding=True,
        return_tensors="pt"
    )
    inputs["target_input_ids"] = targets.input_ids
    inputs["target_attention_mask"] = targets.attention_mask
    inputs.update(labels_dict)

    return inputs