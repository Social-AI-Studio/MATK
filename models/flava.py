import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics


from transformers import FlavaModel

class FlavaClassificationModel(pl.LightningModule):
    def __init__(self, model_class_or_path, num_classes=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = FlavaModel.from_pretrained(model_class_or_path)
        self.mlp = nn.Sequential(
            nn.Linear(self.model.config.multimodal_config.hidden_size, num_classes)
        )

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
    
    def training_step(self, batch, batch_idx):
        labels = batch['labels']

        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        loss = F.cross_entropy(preds, labels)
        
        self.train_acc(preds.argmax(dim=-1), labels)
        self.train_auroc(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_auroc', self.train_auroc, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    
    def validation_step(self, batch, batch_idx):
        labels = batch['labels']

        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        loss = F.cross_entropy(preds, labels)
        
        self.val_acc(preds.argmax(dim=-1), labels)
        self.val_auroc(preds, labels)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_auroc', self.val_auroc, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        
        self.test_acc(preds.argmax(dim=-1), labels)
        self.test_auroc(preds, labels)

        return None

    def on_test_epoch_end(self):
        print("test_acc:", self.test_acc.compute())
        print("test_auroc:", self.test_auroc.compute())

    def predict_step(self, batch, batch_idx):
        model_outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch['pixel_values']
        )
        preds = self.mlp(model_outputs.multimodal_embeddings[:, 0])
        results = {"preds": preds}

        if "labels" in batch:
            results['labels'] = batch["labels"]

        return results
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]