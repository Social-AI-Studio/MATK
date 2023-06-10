import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import RobertaForMaskedLM
from transformers import get_linear_schedule_with_warmup,AdamW
import torchmetrics

def bce_for_loss(logits,labels):
    loss=F.binary_cross_entropy_with_logits(logits, labels)
    loss*=labels.size(1)
    return loss

class RobertaPromptModel(pl.LightningModule):
    def __init__(self, model_class_or_path, label_list):
        super(RobertaPromptModel, self).__init__()
        self.label_word_list=label_list
        self.roberta = RobertaForMaskedLM.from_pretrained(model_class_or_path)

    def forward(self,tokens,attention_mask,mask_pos,feat=None):
        batch_size = tokens.size(0)
        #the position of word for prediction
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
            
        out = self.roberta(tokens, 
                           attention_mask)
        prediction_mask_scores = out.logits[torch.arange(batch_size),
                                          mask_pos]
        
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:,
                                                 self.label_word_list[label_id]
                                                ].unsqueeze(-1))
            #print(prediction_mask_scores[:, self.label_word_list[label_id]].shape)
        logits = torch.cat(logits, -1)
        return logits
        

class PromptModel(pl.LightningModule):
    def __init__(self, model_class_or_path, label_list, opt):
        super().__init__()
        self.save_hyperparameters()
        self.model = RobertaPromptModel(model_class_or_path, label_list)
        self.opt = opt

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(label_list))
        self.train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=len(label_list))

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(label_list))
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=len(label_list))
    
    def training_step(self, batch, batch_idx):
        
        cap = batch['cap_tokens'].long().cuda()
        label = batch['label'].float().cuda().view(-1, 1)
        mask = batch['mask'].cuda()
        target = batch['target'].cuda()
        feat = None
        mask_pos = batch['mask_pos'].cuda()
        logits = self.model(cap, mask, mask_pos, feat)

        if self.opt["FINE_GRIND"]:
            attack = batch['attack'].cuda() #B,6
            logits[:, 1] = torch.sum(logits[:, 1:], dim=1)
            logits = logits[:, :2]

        loss = F.binary_cross_entropy_with_logits(logits, target)
        # loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.train_acc(logits, target.argmax(dim=-1))
        self.train_auroc(logits, target.argmax(dim=-1))
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_auroc', self.train_auroc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        cap = batch['cap_tokens'].long().cuda()
        label = batch['label'].float().cuda().view(-1, 1)
        mask = batch['mask'].cuda()
        target = batch['target'].cuda()
        feat = None
        mask_pos = batch['mask_pos'].cuda()
        logits = self.model(cap, mask, mask_pos, feat)

        if self.opt["FINE_GRIND"]:
            attack = batch['attack'].cuda() #B,6
            logits[:, 1] = torch.sum(logits[:, 1:], dim=1)
            logits = logits[:, :2]

        loss = F.binary_cross_entropy_with_logits(logits, target)
        # loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.val_acc(logits, target.argmax(dim=-1))
        self.val_auroc(logits, target.argmax(dim=-1))
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_auroc', self.val_auroc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        # return [self.optimizer]
        #initialization of optimizer
        params = {}
        for n, p in self.model.named_parameters():
            if self.opt["FIX_LAYERS"] > 0:
                if 'encoder.layer' in n:
                    try:
                        layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                    except:
                        print(n)
                        raise Exception("")
                    if layer_num >= self.opt["FIX_LAYERS"]:
                        print('yes', n)
                        params[n] = p
                    else:
                        print('no ', n)
                elif 'embeddings' in n:
                    print('no ', n)
                else:
                    print('yes', n)
                    params[n] = p
            else:
                params[n] = p
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.opt["WEIGHT_DECAY"],
            },
            {
                "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.opt["LR_RATE"],
            eps=self.opt["EPS"],
        )
        # scheduler is missing
        return [self.optimizer]