import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics


from transformers import FlavaModel

class FlavaClassificationModel(pl.LightningModule):
    def __init__(self, model_class_or_path, num_classes=2):
        super().__init__()
        self.model = FlavaModel.from_pretrained(model_class_or_path)
        self.mlp = nn.Sequential(
            nn.Linear(self.model.config.multimodal_config.hidden_size, num_classes)
        )

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=num_classes)
    
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

        self.val_acc(preds.argmax(dim=-1), labels)
        self.val_auroc(preds, labels)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_auroc', self.val_auroc, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    # def on_train_epoch_end(self):
    #     # Reset metric states after each epoch
    #     self.train_acc.reset()
    #     self.train_auroc.reset()
    
    # def on_validation_epoch_end(self):
    #     print("accuracy: ", self.val_acc.compute())
    #     print("auroc: ", self.val_auroc.compute())

    #     # Reset metric states after each epoch
    #     self.val_acc.reset()
    #     self.val_auroc.reset()
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return [self.optimizer]