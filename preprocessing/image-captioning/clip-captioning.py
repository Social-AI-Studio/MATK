import os
import json
import torch
import tqdm
import clip
import tqdm
import argparse
import numpy as np
from torch import nn

from PIL import Image
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel


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
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(
            prefix).view(-1, self.prefix_length, self.gpt_embedding_size)

        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat,
                       labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size *
                                    prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self

# @title Caption prediction


def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=128, temperature=1., stop_token: str = '.'):

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
            logits = logits[:, -1, :] / \
                (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(
                    1, 0), scores.squeeze(0)
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
                scores_sum_average, next_tokens = scores_sum_average.view(
                    -1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(
                next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + \
                next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)])
                    for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


def main(clip_model_name, clip_weights, tokenizer_name, img_dir, output_dir, device):

    # initialization of basic models: clip and gpt-2
    clip_model, preprocess = clip.load(
        clip_model_name, device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

    # load clip model
    prefix_length = 10
    model = ClipCaptionModel(prefix_length)

    # alter state dict due to the latest changes in transformers
    # https://github.com/rmokady/CLIP_prefix_caption/issues/76
    altered_state_dict = torch.load(clip_weights, map_location=device)
    for i in range(12):
        del altered_state_dict['gpt.transformer.h.' + str(i) + '.attn.bias']
        del altered_state_dict['gpt.transformer.h.' +
                               str(i) + '.attn.masked_bias']

    model.load_state_dict(altered_state_dict)

    model = model.eval()
    model = model.to(device)

    # generating image captions over all images
    files = os.listdir(img_dir)

    # create folder if not exists
    clip_weights_name, _ = os.path.splitext(clip_weights)
    output_dir = os.path.join(output_dir, f"clip_{clip_weights_name}")
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm.tqdm(files, desc=f"Generating captions - {clip_weights_name}"):

        # Check if it exists
        image_name, _ = os.path.splitext(filename)
        output_filepath = os.path.join(output_dir, f"{image_name}.json")
        if os.path.exists(output_filepath):
            continue

        img_path = os.path.join(img_dir, filename)
        file_feat = Image.open(img_path)
        clip_feat = preprocess(file_feat).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(
                clip_feat).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(
                prefix).reshape(1, prefix_length, -1)

        caption = generate_beam(model, tokenizer, embed=prefix_embed)[0]

        # save record
        record = {
            "img": filename,
            "caption": caption
        }

        print(output_filepath)
        with open(output_filepath, "w+") as f:
            f.write(json.dumps(record))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--clip-model-name', type=str, default="ViT-B/32")
    parser.add_argument('--tokenizer-name', type=str, default="gpt2")
    parser.add_argument('--clip-weights', type=str, required=True)
    parser.add_argument('--img-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument("--device", type=str,
                        choices=["cuda", "cpu"], required=True)
    args = parser.parse_args()

    # calculate square
    main(args.clip_model_name, args.clip_weights, args.tokenizer_name,
         args.img_dir, args.output_dir, args.device)
