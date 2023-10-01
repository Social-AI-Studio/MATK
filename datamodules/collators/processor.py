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
    indices_list = [item + "_indices" for item in labels]

    label_index = 0 
    for l, label_indices_list_name in zip(labels, indices_list):
        label_values = [] 
        label_indices = []
        for batch in batches:
            for dataset_item in batch:
                if l in dataset_item: #get index of record from here
                    label_values.append(dataset_item[l])
                    label_indices.append(label_index)
                    label_index+=1
        inputs[l] = torch.tensor(label_values, dtype=torch.int64)
        inputs[label_indices_list_name] = torch.tensor(label_indices, dtype=torch.int64)
    return inputs