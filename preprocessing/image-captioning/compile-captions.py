import os
import json
import argparse

def main(dataset_dir):
    files = os.listdir(dataset_dir)

    d = {}
    for filename in files:
        filepath = os.path.join(dataset_dir, filename)
        
        with open(filepath) as f:
            data = json.load(f)
            
        key, value = data["img"], data["caption"]
        d[key] = value
    
    dataset_dir = os.path.normpath(dataset_dir)
    output_dir = os.path.dirname(dataset_dir)
    basename = os.path.basename(dataset_dir)

    output_filepath = os.path.join(output_dir, f"{basename}.json")
    with open(output_filepath, "w") as f:
        json.dump(d, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Consolidating individual files into a compiled JSON")
    parser.add_argument("--dataset-dir", type=str, required=True)
    args = parser.parse_args()

    main(args.dataset_dir)