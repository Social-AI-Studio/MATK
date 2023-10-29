import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from transformers import VisualBertModel

from datamodules.collators.gqa_lxmert.modeling_frcnn import GeneralizedRCNN
from datamodules.collators.gqa_lxmert.lxmert_utils import Config

from .model_utils import setup_metrics
from .base import BaseLightningModule

class VisualBertClassificationModel(BaseLightningModule):
    def __init__(
            self, 
            model_class_or_path: str,
            frcnn_class_or_path: str,
            freeze_frcnn: bool,
            optimizers: list
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VisualBertModel.from_pretrained(model_class_or_path)
        self.frcnn_class_or_path = frcnn_class_or_path
        self.freeze_frcnn = freeze_frcnn
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
        self.metric_names = [cfg.name.lower() for cfg in metrics_cfg.values()]

        # used for computing overall loss
        self.train_loss = []
        self.val_loss = []
        
    def foward(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        if self.frcnn_class_or_path:
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
        else:
            visual_feats = batch['visual_feats']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_feats,
            visual_attention_mask=torch.ones(visual_feats.shape[:-1]).to('cuda'),
            token_type_ids=token_type_ids,
        )

        loss = 0.0

        for idx, cls_name in enumerate(self.classes):
            indices = batch[f"{cls_name}_indices"]
            
            targets = batch[cls_name]
            logits = self.mlps[idx](outputs.last_hidden_state[indices, 0])

            loss += F.cross_entropy(logits, targets)

            self.compute_metrics_step(
                cls_name, "train", loss, targets, logits)

        return loss / len(self.classes)
    
    def training_step(self, batch, batch_idx):
        loss = self.forward("train", batch)
        self.train_loss.append(loss)

        self.log(f'train_loss', loss, sync_dist=True)
        
    def validation_step(self, batch, batch_idx):
        # this will be triggered during the Trainer's sanity check
        if not hasattr(self, "classes"):
            raise AttributeError("'classes' has not been initialised... Did you forget to call model.setup_tasks()?")

        loss = self.forward("validate", batch)
        self.train_loss.append(loss)

        self.log(f'validate_loss', loss, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        self.forward("test", batch)

    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        if self.frcnn_class_or_path:
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
        else:
            visual_feats = batch['visual_feats']

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeds=visual_feats,
            visual_attention_mask=torch.ones(visual_feats.shape[:-1]).to('cuda'),
            token_type_ids=token_type_ids,
        )

        results = {}
        for idx, cls_name in enumerate(self.classes):
            logits = self.mlps[idx](outputs.last_hidden_state[:, 0, :])
            
            results["img"] = batch["image_filename"].tolist()
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