import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from transformers import LxmertModel

from datamodules.collators.gqa_lxmert.modeling_frcnn import GeneralizedRCNN
from datamodules.collators.gqa_lxmert.lxmert_utils import Config

from .model_utils import setup_metrics, SimpleClassifier
from .base import BaseLightningModule

class LxmertClassificationModel(BaseLightningModule):
    def __init__(
            self, 
            model_class_or_path: str,
            frcnn_class_or_path: str,
            freeze_frcnn: bool,
            dropout: float,
            optimizers: list
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = LxmertModel.from_pretrained(model_class_or_path)
        self.optimizers = optimizers
        self.dropout = dropout

        self.frcnn_class_or_path = frcnn_class_or_path
        self.freeze_frcnn = freeze_frcnn


    def setup_tasks(self, metrics_cfg, cls_cfg):
        # set up the metrics for evaluation
        setup_metrics(self, cls_cfg, metrics_cfg, "train")
        setup_metrics(self, cls_cfg, metrics_cfg, "validate")
        setup_metrics(self, cls_cfg, metrics_cfg, "test")

        # set up the frcnn
        if self.frcnn_class_or_path:
            self.frcnn_cfg = Config.from_pretrained(self.frcnn_class_or_path)
            self.frcnn = GeneralizedRCNN.from_pretrained(self.frcnn_class_or_path, config=self.frcnn_cfg)

            if self.freeze_frcnn:
                for param in self.frcnn.parameters():
                    param.requires_grad = False
        
        # set up metric
        self.mlps = nn.ModuleList([
            SimpleClassifier(
                self.model.config.hidden_size, 
                value, 
                self.dropout
            )
            for value in cls_cfg.values()
        ])
        
        # set up metric
        setup_metrics(self, cls_cfg, metrics_cfg, "train")
        setup_metrics(self, cls_cfg, metrics_cfg, "validate")
        setup_metrics(self, cls_cfg, metrics_cfg, "test")

        self.metric_names = [cfg.name.lower() for cfg in metrics_cfg.values()]
        self.classes = list(cls_cfg.keys())

        # used for computing overall loss
        self.train_loss = []
        self.val_loss = []

    def forward(self, stage, batch):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']

        if self.frcnn_class_or_path:
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
            visual_pos = visual_dict.get("normalized_boxes")
        else:
            visual_feats = batch['visual_feats']
            visual_pos = batch['visual_pos']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_feats=visual_feats,
            visual_pos = visual_pos,
            token_type_ids = token_type_ids
        )

        loss = 0

        for idx, cls_name in enumerate(self.classes):
            indices = batch[f"{cls_name}_indices"]
            targets = batch[cls_name]
            classifier = self.mlps[idx]
            
            logits = classifier(outputs[0][indices, 0])

            loss += F.cross_entropy(logits, targets)

            self.compute_metrics_step(stage, cls_name, targets, logits)

        return loss / len(self.classes)
    
    def training_step(self, batch, batch_idx):
        loss = self.forward("train", batch)
        self.train_loss.append(loss)

        self.log(f'train_loss', loss, sync_dist=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        # this will be triggered during the Trainer's sanity check
        if not hasattr(self, "classes"):
            raise AttributeError("'classes' has not been initialised... Did you forget to call model.setup_tasks()?")

        loss = self.forward("validate", batch)
        self.train_loss.append(loss)

        self.log(f'validate_loss', loss, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        self.forward("test", batch)
        return

    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']

        if self.frcnn_class_or_path:
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
            visual_pos = visual_dict.get("normalized_boxes")
        else:
            visual_feats = batch['visual_feats']
            visual_pos = batch['visual_pos']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_feats=visual_feats,
            visual_pos = visual_pos,
            token_type_ids = token_type_ids
        )

        results = {}
        for idx, cls_name in enumerate(self.classes):
            logits = self.mlps[idx](outputs[0][:, 0])
            
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