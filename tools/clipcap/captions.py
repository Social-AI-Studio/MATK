import argparse
import clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from PIL import Image
import IPython.display as display
import json
import random
import pickle as pkl


def main(clean_img_dir ,model_path, output_dir):

    def read_json(path):
        data=json.load(open(path,'r'))
        return data

    def load_pkl(path):
        data=pkl.load(open(path,'rb'))
        return data

    #set GPU id
    CUDA_DEVICE=1
    torch.cuda.set_device(CUDA_DEVICE)
    device = torch.device("cuda:"+str(CUDA_DEVICE))

    # pretrain_data='conceptual'#or coco
    # save_path = os.path.join(os.path.dirname(''), "pretrained_models")
    # os.makedirs(save_path, exist_ok=True)
    # model_path = os.path.join(save_path, pretrain_data+'_weights.pt')

    N = type(None)
    V = np.array
    ARRAY = np.ndarray
    ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
    VS = Union[Tuple[V, ...], List[V]]
    VN = Union[V, N]
    VNS = Union[VS, N]
    T = torch.Tensor
    TS = Union[Tuple[T, ...], List[T]]
    TN = Optional[T]
    TNS = Union[Tuple[TN, ...], List[TN]]
    TSN = Optional[TS]
    TA = Union[T, ARRAY]

    D = torch.device

    class MLP(nn.Module):

        def forward(self, x: T) -> T:
            return self.model(x)

        def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
            super(MLP, self).__init__()
            layers = []
            for i in range(len(sizes) -1):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
                if i < len(sizes) - 2:
                    layers.append(act())
            self.model = nn.Sequential(*layers)


    class ClipCaptionModel(nn.Module):

        #@functools.lru_cache #FIXME
        def get_dummy_token(self, batch_size: int, device: D) -> T:
            return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

        def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
            embedding_text = self.gpt.transformer.wte(tokens)
            prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
            #print(embedding_text.size()) #torch.Size([5, 67, 768])
            #print(prefix_projections.size()) #torch.Size([5, 1, 768])
            embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
            if labels is not None:
                dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
                labels = torch.cat((dummy_token, tokens), dim=1)
            out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
            return out

        def __init__(self, prefix_length: int, prefix_size: int = 512):
            super(ClipCaptionModel, self).__init__()
            self.prefix_length = prefix_length
            self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
            if prefix_length > 10:  # not enough memory
                self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
            else:
                self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


    class ClipCaptionPrefix(ClipCaptionModel):

        def parameters(self, recurse: bool = True):
            return self.clip_project.parameters()

        def train(self, mode: bool = True):
            super(ClipCaptionPrefix, self).train(mode)
            self.gpt.eval()
            return self

    #@title Caption prediction

    def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                    entry_length=67, temperature=1., stop_token: str = '.'):

        model.eval()
        stop_token_index = tokenizer.encode(stop_token)[0]
        tokens = None
        scores = None
        device = next(model.parameters()).device
        seq_lengths = torch.ones(beam_size, device=device)
        is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
        with torch.no_grad():
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                    generated = model.gpt.transformer.wte(tokens)
            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                logits = logits.softmax(-1).log()
                if scores is None:
                    scores, next_tokens = logits.topk(beam_size, -1)
                    generated = generated.expand(beam_size, *generated.shape[1:])
                    next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                    if tokens is None:
                        tokens = next_tokens
                    else:
                        tokens = tokens.expand(beam_size, *tokens.shape[1:])
                        tokens = torch.cat((tokens, next_tokens), dim=1)
                else:
                    logits[is_stopped] = -float(np.inf)
                    logits[is_stopped, 0] = 0
                    scores_sum = scores[:, None] + logits
                    seq_lengths[~is_stopped] += 1
                    scores_sum_average = scores_sum / seq_lengths[:, None]
                    scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                    next_tokens_source = next_tokens // scores_sum.shape[1]
                    seq_lengths = seq_lengths[next_tokens_source]
                    next_tokens = next_tokens % scores_sum.shape[1]
                    next_tokens = next_tokens.unsqueeze(1)
                    tokens = tokens[next_tokens_source]
                    tokens = torch.cat((tokens, next_tokens), dim=1)
                    generated = generated[next_tokens_source]
                    scores = scores_sum_average * seq_lengths
                    is_stopped = is_stopped[next_tokens_source]
                next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
                if is_stopped.all():
                    break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
        order = scores.argsort(descending=True)
        output_texts = [output_texts[i] for i in order]
        return output_texts


    def generate2(
            model,
            tokenizer,
            tokens=None,
            prompt=None,
            embed=None,
            entry_count=1,
            entry_length=67,  # maximum number of words
            top_p=0.8,
            temperature=1.,
            stop_token: str = '.',
    ):
        model.eval()
        generated_num = 0
        generated_list = []
        stop_token_index = tokenizer.encode(stop_token)[0]
        filter_value = -float("Inf")
        device = next(model.parameters()).device

        with torch.no_grad():

            for entry_idx in trange(entry_count):
                if embed is not None:
                    generated = embed
                else:
                    if tokens is None:
                        tokens = torch.tensor(tokenizer.encode(prompt))
                        tokens = tokens.unsqueeze(0).to(device)

                    generated = model.gpt.transformer.wte(tokens)

                for i in range(entry_length):

                    outputs = model.gpt(inputs_embeds=generated)
                    logits = outputs.logits
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                        ..., :-1
                                                        ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = filter_value
                    next_token = torch.argmax(logits, -1).unsqueeze(0)
                    next_token_embed = model.gpt.transformer.wte(next_token)
                    if tokens is None:
                        tokens = next_token
                    else:
                        tokens = torch.cat((tokens, next_token), dim=1)
                    generated = torch.cat((generated, next_token_embed), dim=1)
                    if stop_token_index == next_token.item():
                        break

                output_list = list(tokens.squeeze().cpu().numpy())
                output_text = tokenizer.decode(output_list)
                generated_list.append(output_text)

        return generated_list[0]
    
    #initialization of basic models: clip and gpt-2
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    prefix_length = 10
    model = ClipCaptionModel(prefix_length)
    model.load_state_dict(torch.load(model_path, map_location=device)) 

    model = model.eval() 
    model = model.to(device)

    use_beam_search = True

    #generating image captions over all images
    files=os.listdir(clean_img_dir)
    total={}
    #random.shuffle(files)
    for i,f in enumerate(files):
        if i%200==0:
            print ('Already finished:',i*100.0/len(files))
        if i == 200:
            break
        img_path=os.path.join(clean_img_dir,f)
        file_feat=Image.open(img_path)
        clip_feat=preprocess(file_feat).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(clip_feat).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        if use_beam_search:
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
        total[f.split('.')[0]]=generated_text_prefix 
    # Create the directory path if it doesn't exist
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Dump the data to the specified path
    with open(output_dir, 'wb') as file:
        pkl.dump(total, file)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # /mnt/sdb/aditi/mmf/data/datasets/memes/defaults/images
    # add arguments
    parser.add_argument('--clean-img-dir', type=str, help='Path to directory containing cleaned images',required=True)
    parser.add_argument('--model-dir', type=str, help='Path to file with pretrained model weights',required=True)
    parser.add_argument('--output-dir', type=str, help='Path that should contained cleaned captions',required=True)
    # parse arguments
    args = parser.parse_args()

    # calculate square
    clean_img_dir = args.clean_img_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    main(clean_img_dir, model_dir, output_dir)

    