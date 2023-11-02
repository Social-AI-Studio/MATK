import torch
import importlib
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .model_utils import setup_metrics
from .base import BaseLightningModule

class T5CLMModel(BaseLightningModule):
    def __init__(
        self,
        model_class_or_path: str,
        optimizers: list
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_class_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_class_or_path, use_fast=True)
        self.optimizers = optimizers

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
                tokens = self.tokenizer.encode(word, add_special_tokens=False)
                self.cls_tokens[cls_name][tokens[0]] = label
        
        # important variables used in the BaseLightningModule
        self.classes = list(cls_cfg.keys())
        self.metric_names = [cfg["name"].lower() for cfg in metrics_cfg.values()]

        # used for computing overall loss
        self.train_loss = []
        self.val_loss = []

    def get_logits(self, outputs, indices, tokens):
        first_word = outputs.logits[indices, 0, :].cpu()

        logits = []
        for token in tokens:
            logits.append(first_word[:,
                                     token
                                     ].unsqueeze(-1))
        logits = torch.cat(logits, -1)
        return logits

    def get_labels(self, labels, token2label):
        targets = [x[0].item() for x in labels]
        targets = [token2label[x] for x in targets]
        return torch.tensor(targets, dtype=torch.int64)
    
    def forward(self, stage, batch):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels_input_ids"]
        )

        for cls_name, token2label in self.cls_tokens.items():
            indices = batch[f"{cls_name}_indices"]
            labels = batch[cls_name]

            logits = self.get_logits(model_outputs, indices, list(token2label.keys()))
            labels = batch[f"{cls_name}"]
            logits, labels = logits.cpu(), labels.cpu()

            self.compute_metrics_step(stage, cls_name, labels, logits)

        return model_outputs.loss / len(self.cls_tokens)

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