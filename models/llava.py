import os
import torch
import torch.nn as nn
import importlib
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration

from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from .model_utils import setup_metrics
from .base import BaseLightningModule

class LlavaCLMModel(BaseLightningModule):
    def __init__(
        self,
        model_class_or_path: str,
        optimizers: list,
        save_dir: str
    ):
        super().__init__()
        self.save_hyperparameters()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(model_class_or_path, quantization_config=bnb_config)
        print(f'Memory used by model: {round(self.model.get_memory_footprint()/1024/1024/1024, 2)} GB')
        LORA_R = 128
        LORA_ALPHA = 256
        
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            
        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, config)
        self.model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(model_class_or_path,use_fast=True)
        self.tokenizer = AutoProcessor.from_pretrained(model_class_or_path,use_fast=True)
        self.tokenizer.tokenizer = tokenizer

        self.optimizers = optimizers
        self.save_dir = save_dir
        

    def setup_tasks(self, metrics_cfg, cls_cfg):
        # set up the metrics for evaluation
        cls_stats = {cls: len(label2word)
                     for cls, label2word in cls_cfg.items()}
        setup_metrics(self, cls_stats, metrics_cfg, "train")
        setup_metrics(self, cls_stats, metrics_cfg, "validate")
        setup_metrics(self, cls_stats, metrics_cfg, "test")

        # set up the token2word conversion for evaluation purposes
        self.cls_tokens = {}
        for cls_name, label2word in cls_cfg.items():
            self.cls_tokens[cls_name] = {}
            for label, word in label2word.items():
                tokens = self.tokenizer.tokenizer.encode(word)
                self.cls_tokens[cls_name][tokens[1]] = label
           
        # important variables used in the BaseLightningModule
        self.classes = list(cls_cfg.keys())
        self.metric_names = [cfg["name"].lower() for cfg in metrics_cfg.values()]

        # used for computing overall loss
        self.train_loss = []
        self.val_loss = []

    def get_logits(self, outputs, indices, tokens):
        first_word = outputs.logits[indices, -1, :].cpu()
        logits = []
        for token in tokens:
            logits.append(first_word[:,
                                     token
                                     ].unsqueeze(-1))
        logits = torch.cat(logits, -1)
        return logits
    
    def forward(self, stage, batch):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        
        loss = 0.0

        for cls_name, token2label in self.cls_tokens.items():
            indices = batch[f"{cls_name}_indices"]
            labels = batch[cls_name]

            logits = self.get_logits(model_outputs, indices, list(token2label.keys())) # some decimal values
            labels = batch[f"{cls_name}"] # 0 or 1

            logits, labels = logits.cpu(), labels.cpu() 
            self.compute_metrics_step(stage, cls_name, labels, logits)
            # explanation = self.tokenizer.batch_decode(torch.argmax(model_outputs.logits[indices, : , :], dim=2).tolist(), skip_special_tokens=True)
            # print(explanation)
            loss += F.cross_entropy(logits, labels)
        return loss / len(self.cls_tokens)

    def training_step(self, batch, batch_idx):
        loss = self.forward("train", batch)
        self.train_loss.append(loss)

        self.log(f'train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this will be triggered during the Trainer's sanity check
        if not hasattr(self, "cls_tokens"):
            raise AttributeError("'cls_tokens' has not been initialised... Did you forget to call model.setup_tasks()?")

        loss = self.forward("validate", batch)
        self.val_loss.append(loss)

        self.log(f'validate_loss', loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.forward("test", batch)
        # remember to not include decoder attention mask?
        return 

    def predict_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        logits = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        results = {"logits": logits}

        if "labels" in batch:
            results['labels'] = batch["labels"]

        return results

    def configure_optimizers(self):
        opts = []
        for opt_cfg in self.optimizers:
            class_name = opt_cfg.pop("class_path")

            package_name = ".".join(class_name.split(".")[:-1])
            package = importlib.import_module(package_name)

            class_name = class_name.split(".")[-1]
            cls = getattr(package, class_name)

            opts.append(cls(self.parameters(), **opt_cfg))

        return opts
    
    def on_save_checkpoint(self, checkpoint):
        # 99% of use cases you don't need to implement this method
        print("Custom saving!!!")
        self.model.save_pretrained(self.save_dir)