import torch
import importlib
import logging
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from transformers import FlavaModel
from .model_utils import setup_metrics


class FlavaClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_class_or_path: str,
        metrics_cfg: dict,
        cls_dict: dict,
        optimizers: list
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = FlavaModel.from_pretrained(model_class_or_path)
        self.metric_names = [cfg.name.lower() for cfg in metrics_cfg.values()]
        self.cls_dict = cls_dict
        self.optimizers = optimizers

        # set up classification
        self.mlps = nn.ModuleList([
            nn.Linear(self.model.config.multimodal_config.hidden_size, num_classes)
            for num_classes in cls_dict.values()
        ])

        # set up metric
        setup_metrics(self, cls_dict, metrics_cfg, "train")
        setup_metrics(self, cls_dict, metrics_cfg, "validate")
        setup_metrics(self, cls_dict, metrics_cfg, "test")

    def compute_metrics_step(self, cls_name, stage, loss, targets, preds):
        self.log(f'{stage}_{cls_name}_loss', loss, prog_bar=True)

        for metric_name in self.metric_names:
            metric = getattr(self, f"{stage}_{cls_name}_{metric_name}")
            metric(preds, targets)

    def compute_metrics_epoch(self, cls_name, stage):
        msg = "Epoch Results:\n"

        avg_metric_score = 0
        for metric_name in self.metric_names:
            metric = getattr(self, f"{stage}_{cls_name}_{metric_name}")
            metric_score = metric.compute()
            avg_metric_score += metric_score

            self.log(f'{stage}_{cls_name}_{metric_name}', metric_score,
                     prog_bar=True, sync_dist=True)
            
            msg += f"\t{stage}_{cls_name}_{metric_name}: {metric_score}\n"

            # reset the metrics
            metric.reset()


        self.log(f'{stage}_{cls_name}_average', avg_metric_score,
                    prog_bar=True, sync_dist=True)
        
        avg_metric_score = avg_metric_score / len(self.metric_names)
        msg += f"\t{stage}_{cls_name}_average: {avg_metric_score}\n"

        logging.info(msg)


    def training_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        total_loss = 0.0
        
        for idx, cls_name in enumerate(self.cls_dict.keys()):
            targets = batch[cls_name]
            preds = self.mlps[idx](model_outputs.multimodal_embeddings[:, 0])

            loss = F.cross_entropy(preds, targets)
            total_loss += loss
            
            self.compute_metrics_step(
                cls_name, "train", loss, targets, preds)


        return total_loss / len(self.cls_dict)

    def on_training_epoch_end(self):
        for cls_name in self.cls_dict.keys():
            self.compute_metrics_epoch(cls_name, "train")

    def validation_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        total_loss = 0.0

        for idx, cls_name in enumerate(self.cls_dict.keys()):
            targets = batch[cls_name]
            preds = self.mlps[idx](model_outputs.multimodal_embeddings[:, 0])

            loss = F.cross_entropy(preds, targets)
            total_loss += loss

            self.compute_metrics_step(
                cls_name, "validate", loss, targets, preds)

            total_loss += loss

        return total_loss / len(self.cls_dict)

    def on_validation_epoch_end(self):
        for cls_name in self.cls_dict.keys():
            self.compute_metrics_epoch(cls_name, "validate")

    def test_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        total_loss = 0.0

        for idx, cls_name in enumerate(self.cls_dict.keys()):
            targets = batch[cls_name]
            preds = self.mlps[idx](model_outputs.multimodal_embeddings[:, 0])

            loss = F.cross_entropy(preds, targets)
            total_loss += loss

            self.compute_metrics_step(
                cls_name, "test", loss, targets, preds)

            total_loss += loss

        return total_loss / len(self.cls_dict)

    def on_test_epoch_end(self):
        for cls_name in self.cls_dict.keys():
            self.compute_metrics_epoch(cls_name, "test")

    def predict_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        results = {}
        for idx, cls_name in enumerate(self.cls_dict.keys()):
            label_preds = self.mlps[idx](
                model_outputs.multimodal_embeddings[:, 0])
            
            results[cls_name] = label_preds

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
