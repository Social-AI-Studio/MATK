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
    
    for l in labels:
        label_values = [] 
        for batch in batches:
            for dataset_item in batch:
                if l in dataset_item:
                    label_values.append(dataset_item[l])
        
        inputs[l] = torch.tensor(label_values, dtype=torch.int64)
    return inputs