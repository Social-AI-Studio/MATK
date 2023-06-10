import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import lightning.pytorch as pl
from transformers import RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup,AdamW
from model_utils.prompthate.classifier import SingleClassifier, SimpleClassifier
from model_utils.prompthate.rela_encoder import Rela_Module 

class RobertaBaseModel(nn.Module):
    def __init__(self,roberta,classifier,attention,proj_v):
        super(RobertaBaseModel, self).__init__()
        self.text_encoder=roberta
        self.classifier=classifier
        self.attention=attention
        self.proj_v=proj_v

    def forward(self,tokens,attention_mask,feat=None):
        output=self.text_encoder(tokens,
                                 attention_mask=attention_mask)[1][-1]
        if feat==None:
            joint_repre=output[:,0]
        else:
            #print ('Multimodal')
            text_repre=output[:,0]
            vis=self.proj_v(feat)
            att_vis=self.attention(vis,output)
            joint_repre=torch.cat((att_vis,text_repre),dim=1)
        logits=self.classifier(joint_repre)
        return logits
        
class BaseModel(pl.LightningModule):
    def __init__(self, model_class_or_path, opt):
        super().__init__()
        self.save_hyperparameters()
        final_dim=2
        times=2-int(opt["UNIMODAL"])
        text_encoder=RobertaForSequenceClassification.from_pretrained(
            model_class_or_path,
            num_labels=final_dim,
            output_attentions=False,
            output_hidden_states=True
        )
        attention=Rela_Module(opt["ROBERTA_DIM"],
                            opt["ROBERTA_DIM"],opt["NUM_HEAD"],opt["MID_DIM"],
                            opt["TRANS_LAYER"],
                            opt["FC_DROPOUT"])
        classifier=SimpleClassifier(opt["ROBERTA_DIM"]*times,
                                    opt["MID_DIM"],final_dim,opt["FC_DROPOUT"])
        proj_v=SingleClassifier(opt["FEAT_DIM"],opt["ROBERTA_DIM"],opt["FC_DROPOUT"])
        self.model = RobertaBaseModel(text_encoder,classifier,attention,proj_v)
       
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=2)

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=2)

    def training_step(self, batch, batch_idx):
        cap=batch['cap_tokens'].long().cuda()
        label=batch['label'].float().cuda().view(-1,1)
        mask=batch['mask'].cuda()
        target=batch['target'].cuda()
        feat=None
        logits=self.model(cap,mask,feat)

        loss = F.cross_entropy(logits, target)
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
        cap=batch['cap_tokens'].long().cuda()
        label=batch['label'].float().cuda().view(-1,1)
        mask=batch['mask'].cuda()
        target=batch['target'].cuda()
        feat=None
        logits=self.model(cap,mask,feat)

        loss = F.cross_entropy(logits, target)

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
        params=self.model.parameters()
        self.optimizer=AdamW(params,
                    lr=1e-5,
                    eps=1e-8
                   )
        return [self.optimizer]
    