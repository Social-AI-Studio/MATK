import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from transformers import VisualBertModel

from datamodules.collators.gqa_lxmert.modeling_frcnn import GeneralizedRCNN
from datamodules.collators.gqa_lxmert.lxmert_utils import Config

from .model_utils import setup_metrics, collapse_cls_dict
from .base import BaseLightningModule

class VisualBertClassificationModel(BaseLightningModule):
    def __init__(
            self, 
            model_class_or_path: str,
            frcnn_class_or_path: str,
            frcnn_trainable: bool,
            metrics_cfg: dict,
            cls_dict: dict,
            optimizers: list
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VisualBertModel.from_pretrained(model_class_or_path)
        self.metric_names = [cfg.name.lower() for cfg in metrics_cfg.values()]
        self.optimizers = optimizers

        if frcnn_class_or_path:
            self.frcnn_cfg = Config.from_pretrained(frcnn_class_or_path)
            self.frcnn = GeneralizedRCNN.from_pretrained(frcnn_class_or_path, config=self.frcnn_cfg)

            if not frcnn_trainable:
                for param in self.frcnn.parameters():
                    param.requires_grad = False

        collapsed_label_dict = collapse_cls_dict(cls_dict)
        self.classes = list(collapsed_label_dict)

        # set up classification
        self.mlps = nn.ModuleList([
            nn.Linear(self.model.config.hidden_size, value)
            for value in collapsed_label_dict.values()
        ])
        
        # set up metric
        setup_metrics(self, collapsed_label_dict, metrics_cfg, "train")
        setup_metrics(self, collapsed_label_dict, metrics_cfg, "validate")
        setup_metrics(self, collapsed_label_dict, metrics_cfg, "test")
        
    def training_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        if "visual_feats" in batch and "visual_pos" in batch:
            visual_feats = batch['visual_feats']
        else:
            # Run Faster-RCNN
            images = batch['images']
            sizes = batch['sizes']
            scales_yx = batch['scales_yx']

            visual_dict = self.frcnn(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=self.frcnn_cfg.max_detections,
                return_tensors="pt",
                location=images.device
            )

            visual_feats = visual_dict['roi_features']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_feats,
            visual_attention_mask=torch.ones(visual_feats.shape[:-1]).to('cuda'),
            token_type_ids=token_type_ids,
        )

        total_loss = 0
        batch_start_index = 0
        batch_end_index = 0
        for idx, cls_name in enumerate(self.classes):
            targets = batch[cls_name]
            batch_end_index += len(targets)
            preds = self.mlps[idx](outputs.last_hidden_state[batch_start_index: batch_end_index, :][:, 0, :])
            loss = F.cross_entropy(preds, targets)
            total_loss += loss
            batch_start_index =batch_end_index

            self.compute_metrics_step(
                cls_name, "train", loss, targets, preds)

        return total_loss / len(self.classes)
    
    def validation_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        if "visual_feats" in batch and "visual_pos" in batch:
            visual_feats = batch['visual_feats']
        else:
            # Run Faster-RCNN
            images = batch['images']
            sizes = batch['sizes']
            scales_yx = batch['scales_yx']
            
            visual_dict = self.frcnn(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=self.frcnn_cfg.max_detections,
                return_tensors="pt",
                location=images.device
            )
            
            visual_feats = visual_dict['roi_features']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_feats,
            visual_attention_mask=torch.ones(visual_feats.shape[:-1]).to('cuda'),
            token_type_ids=token_type_ids,
        )

        total_loss = 0
        batch_start_index = 0
        batch_end_index = 0
        for idx, cls_name in enumerate(self.classes):
            targets = batch[cls_name]
            batch_end_index += len(targets)
            preds = self.mlps[idx](outputs.last_hidden_state[batch_start_index: batch_end_index, :][:, 0, :])
            loss = F.cross_entropy(preds, targets)
            total_loss += loss
            batch_start_index =batch_end_index
            self.compute_metrics_step(
                cls_name, "validate", loss, targets, preds)

        return total_loss / len(self.classes)
    
    def test_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        if "visual_feats" in batch and "visual_pos" in batch:
            visual_feats = batch['visual_feats']
        else:
            # Run Faster-RCNN
            images = batch['images']
            sizes = batch['sizes']
            scales_yx = batch['scales_yx']
            
            visual_dict = self.frcnn(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=self.frcnn_cfg.max_detections,
                return_tensors="pt",
                location=images.device
            )

            visual_feats = visual_dict['roi_features']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_feats,
            visual_attention_mask=torch.ones(visual_feats.shape[:-1]).to('cuda'),
            token_type_ids=token_type_ids,
        )

        total_loss = 0
        batch_start_index = 0
        batch_end_index = 0
        for idx, cls_name in enumerate(self.classes):
            targets = batch[cls_name]
            batch_end_index += len(targets)
            preds = self.mlps[idx](outputs.last_hidden_state[batch_start_index: batch_end_index, :][:, 0, :])
            loss = F.cross_entropy(preds, targets)
            total_loss += loss
            batch_start_index =batch_end_index
            self.compute_metrics_step(
                cls_name, "test", loss, targets, preds)
        
        return total_loss / len(self.classes)

    def predict_step(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        if "visual_feats" in batch and "visual_pos" in batch:
            visual_feats = batch['visual_feats']
        else:
            # Run Faster-RCNN
            images = batch['images']
            sizes = batch['sizes']
            scales_yx = batch['scales_yx']
            
            visual_dict = self.frcnn(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=self.frcnn_cfg.max_detections,
                return_tensors="pt",
                location=images.device
            )

            visual_feats = visual_dict['roi_features']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_feats,
            visual_attention_mask=torch.ones(visual_feats.shape[:-1]).to('cuda'),
            token_type_ids=token_type_ids,
        )

        results = {}
        for idx, cls_name in enumerate(self.classes):
            preds = self.mlps[idx](outputs.last_hidden_state[:, 0, :])
            
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