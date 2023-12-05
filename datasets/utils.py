import json
import csv

def _load_jsonl(filepath):
    data = []
    
    with open(filepath, 'r')as f:
        for line in f.readlines():
            obj = json.loads(line)
            data.append(obj)
    
    return data

def _load_csv(file_path): 
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data


def _load_tsv(filepath):
    data = [] 
    with open(filepath, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            data.append(row)
    return data