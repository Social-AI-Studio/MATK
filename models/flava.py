import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F

from transformers import FlavaModel
from .model_utils import setup_metrics, SimpleClassifier
from .base import BaseLightningModule


class FlavaClassificationModel(BaseLightningModule):
    def __init__(
        self,
        model_class_or_path: str,
        dropout: float,
        optimizers: list
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = FlavaModel.from_pretrained(model_class_or_path)
        self.dropout = dropout
        self.optimizers = optimizers

    def setup_tasks(self, metrics_cfg, cls_cfg):
        # set up the metrics for evaluation
        setup_metrics(self, cls_cfg, metrics_cfg, "train")
        setup_metrics(self, cls_cfg, metrics_cfg, "validate")
        setup_metrics(self, cls_cfg, metrics_cfg, "test")

        # set up the various classification heads
        self.mlps = nn.ModuleList([
            SimpleClassifier(
                self.model.config.hidden_size,
                num_classes,
                self.dropout
            )
            for num_classes in cls_cfg.values()
        ])

        # important variables used in the BaseLightningModule
        self.classes = list(cls_cfg.keys())
        self.metric_names = [cfg.name.lower() for cfg in metrics_cfg.values()]

        # used for computing overall loss
        self.train_loss = []
        self.val_loss = []

    def forward(self, stage, batch):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        loss = 0.0

        for idx, cls_name in enumerate(self.classes):
            indices = batch[f"{cls_name}_indices"]
            targets = batch[cls_name]
            classifier = self.mlps[idx]

            logits = classifier(
                model_outputs.multimodal_embeddings[indices, 0]
            )

            loss += F.cross_entropy(logits, targets)

            self.compute_metrics_step(stage, cls_name, targets, logits)

        return loss / len(self.classes)

    def training_step(self, batch, batch_idx):
        loss = self.forward("train", batch)
        self.train_loss.append(loss)

        self.log(f'train_loss', loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # this will be triggered during the Trainer's sanity check
        if not hasattr(self, "classes"):
            raise AttributeError(
                "'classes' has not been initialised... Did you forget to call model.setup_tasks()?")

        loss = self.forward("validate", batch)
        self.train_loss.append(loss)

        self.log(f'validate_loss', loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        self.forward("test", batch)

    def predict_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        results = {}
        for idx, cls_name in enumerate(self.classes):
            logits = self.mlps[idx](model_outputs.multimodal_embeddings[:, 0])
            results[f"{cls_name}_preds"] = torch.argmax(logits, dim=1).tolist()

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
