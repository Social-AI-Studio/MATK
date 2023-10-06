import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .model_utils import setup_metrics
from .base import BaseLightningModule


class FlanT5ClassificationModel(BaseLightningModule):
    def __init__(
        self,
        model_class_or_path: str,
        metrics_cfg: dict,
        cls_labels: dict,
        optimizers: list
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_class_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_class_or_path, use_fast=False)
        self.metric_names = [cfg.name.lower() for cfg in metrics_cfg.values()]
        self.classes = list(cls_labels.keys())
        self.optimizers = optimizers

        # set up metric
        cls_stats = {cls: len(label2word)
                     for cls, label2word in cls_labels.items()}
        setup_metrics(self, cls_stats, metrics_cfg, "train")
        setup_metrics(self, cls_stats, metrics_cfg, "validate")
        setup_metrics(self, cls_stats, metrics_cfg, "test")

        self.cls_tokens = {}
        for cls_name, label2word in cls_labels.items():
            self.cls_tokens[cls_name] = {}
            for label, word in label2word.items():
                tokens = self.tokenizer.encode(word, add_special_tokens=False)
                self.cls_tokens[cls_name][tokens[0]] = label

                assert len(tokens) == 1
        # print(self.cls_tokens) # {'label': {150: 0, 4273: 1}}

    def get_logits(self, outputs, tokens):
        first_word = outputs.logits[:, 1, :].cpu()

        logits = []
        for token in tokens:
            logits.append(first_word[:,
                                     token
                                     ].unsqueeze(-1))
        logits = torch.cat(logits, -1)
        return logits
    
    def get_labels(self, labels, token2label):
        # print(labels) # tensor([[4273,    1]], device='cuda:0')
        # print(token2label) # {150: 0, 4273: 1}
        targets = [x[0].item() for x in labels]
        targets = [token2label[x] for x in targets]
        return torch.tensor(targets, dtype=torch.int64)

    def training_step(self, batch, batch_idx):
        total_loss = 0.0

        for cls_name, token2label in self.cls_tokens.items():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch[cls_name]

            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += model_outputs.loss

            preds = self.get_logits(model_outputs, list(token2label.keys()))
            labels = self.get_labels(labels, token2label)
            preds = preds.to(input_ids.device)
            labels = labels.to(input_ids.device)

            self.compute_metrics_step(
                cls_name, "train", model_outputs.loss, labels, preds)

        return total_loss / len(self.cls_tokens)

    def validation_step(self, batch, batch_idx):
        total_loss = 0.0

        for cls_name, token2label in self.cls_tokens.items(): # {'label': {150: 0, 4273: 1}}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch[cls_name]

            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += model_outputs.loss
            preds = self.get_logits(model_outputs, list(token2label.keys()))
            labels = self.get_labels(labels, token2label)
            preds = preds.to(input_ids.device)
            labels = labels.to(input_ids.device)

            self.compute_metrics_step(
                cls_name, "validate", model_outputs.loss, labels, preds)

        return total_loss / len(self.cls_tokens)

    def test_step(self, batch, batch_idx):
        total_loss = 0.0

        for cls_name, token2label in self.cls_tokens.items():
            labels = batch[cls_name]
            
            model_outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=labels
            )
            total_loss += model_outputs.loss

            preds = self.get_logits(model_outputs, list(token2label.keys()))
            labels = self.get_labels(labels, token2label)
            # preds, labels = preds.cpu(), labels.cpu()

            self.compute_metrics_step(
                cls_name, "test", model_outputs.loss, labels, preds)

        return total_loss / len(self.cls_tokens)

    def predict_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        results = {"preds": preds}

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