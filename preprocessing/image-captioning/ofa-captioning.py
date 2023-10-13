from PIL import Image
from torchvision import transforms
from transformers import OFAModel, OFATokenizer
from transformers.models.ofa.generate import sequence_generator
import torch
import argparse
import os
import pickle as pkl

def main(clean_img_dir, model_dir, output_dir, device):

    device = torch.device

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 480
    patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])

    tokenizer = OFATokenizer.from_pretrained(model_dir)

    txt = " what does the image describe?"
    inputs = tokenizer([txt], return_tensors="pt").input_ids


    model = OFAModel.from_pretrained(model_dir, use_cache=False)
    model.to(device)
    generator = sequence_generator.SequenceGenerator(
                        tokenizer=tokenizer,
                        beam_size=5,
                        max_len_b=16, 
                        min_len=0,
                        no_repeat_ngram_size=3,
                    )
    generator.to(device)    

    files=os.listdir(clean_img_dir)
    total={}
    
    for i,f in enumerate(files):
        if i%200==0:
            print ('Already finished:',i*100.0/len(files))
        img_path=os.path.join(clean_img_dir,f)
        img = Image.open(img_path)
        patch_img = patch_resize_transform(img).unsqueeze(0).to(device) 

        data = {}
        data["net_input"] = {"input_ids": inputs.to(device), 'patch_images': patch_img, 'patch_masks':torch.tensor([True]).to(device)}
        gen_output = generator.generate([model], data)
        gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

        gen = model.generate(inputs.to(device), patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)
        img_caption = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        total[f.split('.')[0]]=img_caption

    # Create the directory path if it doesn't exist
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Dump the data to the specified path
    with open(output_dir, 'wb') as file:
        pkl.dump(total, file)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--clean-img-dir', type=str, help='Path to directory containing cleaned images',required=True)
    parser.add_argument('--model-dir', type=str, help='Path to file with pretrained model weights',required=True)
    parser.add_argument('--output-dir', type=str, help='Path that should contained cleaned captions',required=True)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], required=True)
    # parse arguments
    args = parser.parse_args()

    
    main(clean_img_dir, model_dir, output_dir, device)
