import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F

from transformers import FlavaModel
from .model_utils import setup_metrics, collapse_cls_dict
from .base import BaseLightningModule


class FlavaClassificationModel(BaseLightningModule):
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
        self.optimizers = optimizers
        
        collapsed_label_dict = collapse_cls_dict(cls_dict)
        self.classes = list(collapsed_label_dict) ## fhm_label, mami_misogyny .....

        self.mlps = nn.ModuleList([
            nn.Linear(self.model.config.hidden_size, value)
            for value in collapsed_label_dict.values()
        ])

        # set up metric
        setup_metrics(self, collapsed_label_dict, metrics_cfg, "train")
        setup_metrics(self, collapsed_label_dict, metrics_cfg, "validate")
        setup_metrics(self, collapsed_label_dict, metrics_cfg, "test")


    def training_step(self, batch, batch_idx): 
        
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        total_loss = 0.0
        batch_end_index = 0
        batch_start_index = 0
        for idx, cls_name in enumerate(self.classes):
            targets = batch[cls_name]
            batch_end_index += len(targets)
            preds = self.mlps[idx]((model_outputs.multimodal_embeddings[batch_start_index:batch_end_index, :][:, 0]))
            loss = F.cross_entropy(preds, targets)
            total_loss += loss
            batch_start_index = batch_end_index
            self.compute_metrics_step(cls_name, "train", loss, targets, preds)

        return total_loss / len(self.classes)

    def validation_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
       
        total_loss = 0.0
        batch_end_index = 0
        batch_start_index = 0
        for idx, cls_name in enumerate(self.classes):
            targets = batch[cls_name]
            batch_end_index += len(targets)
            preds = self.mlps[idx]((model_outputs.multimodal_embeddings[batch_start_index:batch_end_index, :][:, 0]))
            loss = F.cross_entropy(preds, targets)
            total_loss += loss
            batch_start_index = batch_end_index
            self.compute_metrics_step(cls_name, "validate", loss, targets, preds)

    def test_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        total_loss = 0.0
        batch_end_index = 0
        batch_start_index = 0
        for idx, cls_name in enumerate(self.classes):
            targets = batch[cls_name]
            batch_end_index += len(targets)
            preds = self.mlps[idx]((model_outputs.multimodal_embeddings[batch_start_index:batch_end_index, :][:, 0]))
            loss = F.cross_entropy(preds, targets)
            total_loss += loss
            batch_start_index = batch_end_index
            
            self.compute_metrics_step(cls_name, "test", loss, targets, preds)

        return total_loss / len(self.classes)

    def predict_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        results = {}
        for idx, cls_name in enumerate(self.classes):
            preds = self.mlps[idx](model_outputs.multimodal_embeddings[:, 0])
            
            results["img"] = batch["image_filename"].tolist()
            results[f"{cls_name}_preds"] = torch.argmax(preds, dim=1).tolist()
            results[f"{cls_name}_labels"] = batch[cls_name].tolist()

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
