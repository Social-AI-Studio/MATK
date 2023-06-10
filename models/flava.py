import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import torchmetrics

from transformers import FlavaModel


class FlavaClassificationModel(pl.LightningModule):
    def __init__(
            self, 
            model_class_or_path: str,
            cls_dict: dict
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = FlavaModel.from_pretrained(model_class_or_path)

        # set up classification
        self.mlps = nn.ModuleList([
            nn.Linear(self.model.config.multimodal_config.hidden_size, value) 
            for value in cls_dict.values()
        ])
        
        # set up metric
        self.cls_dict = cls_dict
        for stage in ["train", "validate", "test"]:
            for key, value in cls_dict.items():
                setattr(self, f"{key}_{stage}_acc", torchmetrics.Accuracy(task="multiclass", num_classes=value))
                setattr(self, f"{key}_{stage}_auroc", torchmetrics.AUROC(task="multiclass", num_classes=value))
       

    def compute_metrics_and_logs(self, cls_name, stage, loss, targets, preds):
        accuracy_metric = getattr(self, f"{cls_name}_{stage}_acc")
        auroc_metric = getattr(self, f"{cls_name}_{stage}_auroc")

        accuracy_metric(preds.argmax(dim=-1), targets)
        auroc_metric(preds, targets)

        self.log(f'{cls_name}_{stage}_loss', loss, prog_bar=True)
        self.log(f'{cls_name}_{stage}_acc', accuracy_metric, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{cls_name}_{stage}_auroc', auroc_metric, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


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
            self.compute_metrics_and_logs(cls_name, "train", loss, targets, preds)

            total_loss += loss
        
        return total_loss / len(self.cls_dict)
    
    
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
            self.compute_metrics_and_logs(cls_name, "validate", loss, targets, preds)

            total_loss += loss
        
        return total_loss / len(self.cls_dict)

    def predict_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )

        results = {}
        for idx, cls_name in enumerate(self.cls_dict.keys()):
            label_preds = self.mlps[idx](model_outputs.multimodal_embeddings[:, 0])
            results[cls_name] = label_preds

        if "labels" in batch:
            results['labels'] = batch["labels"]

        return results
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]