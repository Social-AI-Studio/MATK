import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics

from transformers import BartForConditionalGeneration, AutoTokenizer

class BARTCasualModel(pl.LightningModule):
    def __init__(self, model_class_or_path, label2word: dict):
        super().__init__()
        self.save_hyperparameters()
        self.model = BartForConditionalGeneration.from_pretrained(model_class_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_class_or_path, use_fast=False)

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(label2word))
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=len(label2word))

        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(label2word))
        self.test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=len(label2word))

        self.token2label = {}
        for label, word in label2word.items():
            token = self.tokenizer.encode(word, add_special_tokens=False)[0]
            self.token2label[token] = label
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log('train_loss', outputs.loss, prog_bar=True)
        return outputs.loss
    
    
    def validation_step(self, batch, batch_idx):
        labels = batch['labels']

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels = batch['labels']
        )

        first_word = outputs.logits[:,0,:].cpu()
        labels = labels.cpu()
        
        logits = []
        for token in self.token2label.keys():
            logits.append(first_word[:,
                                   token
                                  ].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        preds = logits.argmax(dim=-1)
        targets = [x[0].item() for x in labels]
        targets = [self.token2label[x] for x in targets]
        targets = torch.tensor(targets, dtype=torch.int64)
        
        self.val_acc(preds, targets)
        self.val_auroc(logits, targets)
        
        self.log('val_loss', outputs.loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_auroc', self.val_auroc, on_step=True, on_epoch=True, sync_dist=True)

        return outputs.loss
    
    def test_step(self, batch, batch_idx):
        labels = batch['labels']

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels = batch['labels']
        )

        first_word = outputs.logits[:,0,:].cpu()
        labels = labels.cpu()
        
        logits = []
        for token in self.token2label.keys():
            logits.append(first_word[:,
                                   token
                                  ].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        preds = logits.argmax(dim=-1)
        targets = [x[0].item() for x in labels]
        targets = [self.token2label[x] for x in targets]
        targets = torch.tensor(targets, dtype=torch.int64)
        
        self.test_acc(preds, targets)
        self.test_auroc(logits, targets)

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
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return [self.optimizer]