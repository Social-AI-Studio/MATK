import torch
import importlib
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_utils import setup_metrics, setup_generation_metrics
from .base import BaseLightningModule

class MistralCLMModel(BaseLightningModule):
    def __init__(
        self,
        model_class_or_path: str,
        optimizers: list
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_class_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_class_or_path, use_fast=True)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.optimizers = optimizers

    def setup_tasks(self, metrics_cfg, cls_cfg):
        # set up the metrics for evaluation
        setup_generation_metrics(self, metrics_cfg, "train")
        setup_generation_metrics(self, metrics_cfg, "validate")
        setup_generation_metrics(self, metrics_cfg, "test")
        
        # important variables used in the BaseLightningModule
        self.classes = ["target"]
        self.metric_names = [cfg["name"].lower() for cfg in metrics_cfg.values()]

        # used for computing overall loss
        self.train_loss = []
        self.val_loss = []
    
    def forward(self, stage, batch):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["target_input_ids"]
        )

        preds = self.tokenizer.batch_decode(torch.argmax(model_outputs.logits, dim=2).tolist(), skip_special_tokens=True)
        targets = self.tokenizer.batch_decode(batch["target_input_ids"], skip_special_tokens=True)
        self.compute_metrics_step(stage, "target", targets, preds)
        return model_outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self.forward("train", batch)
        self.train_loss.append(loss)

        self.log(f'train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this will be triggered during the Trainer's sanity check
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