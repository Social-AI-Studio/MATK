import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
import torchmetrics

from transformers import LxmertModel

from datamodules.collators.gqa_lxmert.modeling_frcnn import GeneralizedRCNN
from datamodules.collators.gqa_lxmert.lxmert_utils import Config


class LxmertClassificationModel(pl.LightningModule):
    def __init__(self, 
                 model_class_or_path, 
                 frcnn_class_or_path,
                 cls_dict):
        super().__init__()
        self.save_hyperparameters()

        self.model = LxmertModel.from_pretrained(model_class_or_path)
        if frcnn_class_or_path:
            self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
            self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)

        # set up classification
        self.mlps = nn.ModuleList([
            nn.Linear(self.model.config.hidden_size, value) 
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

        input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']
        attention_mask = batch['attention_mask']

        if hasattr(self, "frcnn"):
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
            )

            visual_feats = visual_dict['visual_feats']
            visual_pos = visual_dict['visual_pos']
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

        label_list = []
        for k,v in self.cls_dict.items():
            label_list.append(k)

        for i in range(len(self.cls_dict)):
            label_targets = batch[label_list[i]]
            pooled_output = outputs[0][:, 0]  # Extract the [CLS] token embedding
            label_preds = self.mlps[i](pooled_output)
            label_loss = F.cross_entropy(label_preds, label_targets)
            loss += label_loss
            self.compute_metrics_and_logs(label_list[i], "train", label_loss, label_targets, label_preds)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']

        if "visual_feats" in batch and "visual_pos" in batch:
            visual_feats = batch['visual_feats']
            visual_pos = batch['visual_pos']
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
            )

            visual_feats = visual_dict['visual_feats']
            visual_pos = visual_dict['visual_pos']
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_feats=visual_feats,
            visual_pos = visual_pos,
            token_type_ids = token_type_ids
        )


        loss = 0

        label_list = []
        for k,v in self.cls_dict.items():
            label_list.append(k)

        for i in range(len(self.cls_dict)):
            label_targets = batch[label_list[i]]
            pooled_output = outputs[0][:, 0]  # Extract the [CLS] token embedding
            label_preds = self.mlps[i](pooled_output)
            label_loss = F.cross_entropy(label_preds, label_targets)
            loss += label_loss
            self.compute_metrics_and_logs(label_list[i], "validate", label_loss, label_targets, label_preds)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        visual_feats = batch['visual_feats']
        visual_pos = batch['visual_pos']
        token_type_ids = batch['token_type_ids']
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_feats=visual_feats,
            visual_pos = visual_pos,
            token_type_ids = token_type_ids
        )


        loss = 0

        label_list = []
        for k,v in self.cls_dict.items():
            label_list.append(k)

        for i in range(len(self.cls_dict)):
            label_targets = batch[label_list[i]]
            pooled_output = outputs[0][:, 0]  # Extract the [CLS] token embedding
            label_preds = self.mlps[i](pooled_output)
            label_loss = F.cross_entropy(label_preds, label_targets)
            loss += label_loss
            self.compute_metrics_and_logs(label_list[i], "test", label_loss, label_targets, label_preds)
        
        return loss

    def on_test_epoch_end(self):
        print("test_acc:", self.test_acc.compute())
        print("test_auroc:", self.test_auroc.compute())

    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        visual_feats = batch['visual_feats']
        visual_pos = batch['visual_pos']
        token_type_ids = batch['token_type_ids']
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_feats=visual_feats,
            visual_pos = visual_pos,
            token_type_ids = token_type_ids
        )


        loss = 0

        label_list = []
        for k,v in self.cls_dict.items():
            label_list.append(k)

        results = {}
        for i in range(len(self.cls_dict)):
            label_targets = batch[label_list[i]]
            pooled_output = outputs[0][:, 0]  # Extract the [CLS] token embedding
            label_preds = self.mlps[i](pooled_output)
            results[label_list[i]] = label_preds
        
        if "labels" in batch:
            results['labels'] = batch["labels"]

        return results
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]