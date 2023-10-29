import torch
import importlib

import torch.nn.functional as F
import lightning.pytorch as pl

from transformers import RobertaForMaskedLM
from .model_utils import setup_metrics

class RobertaPromptModel(pl.LightningModule):
    def __init__(self, model_class_or_path, label_list):
        super(RobertaPromptModel, self).__init__()
        self.label_word_list = label_list
        self.base_model = RobertaForMaskedLM.from_pretrained(model_class_or_path)

    def forward(self, tokens, attention_mask, mask_pos):
        batch_size = tokens.size(0)

        #the position of word for prediction
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
            
        out = self.base_model(
            tokens, 
            attention_mask
        )

        prediction_mask_scores = out.logits[
            torch.arange(batch_size),
            mask_pos
        ]
        
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[
                :,
                self.label_word_list[label_id]
            ].unsqueeze(-1))
            
        logits = torch.cat(logits, -1)
        return logits
        

class PromptModel(pl.LightningModule):
    def __init__(self, model_class_or_path, label_list, opt):
        super().__init__()
        self.save_hyperparameters()
        self.model = RobertaPromptModel(model_class_or_path, label_list)
        self.opt = opt

    def setup_tasks(self, metrics_cfg, cls_cfg):
        # set up metric
        setup_metrics(self, cls_cfg, metrics_cfg, "train")
        setup_metrics(self, cls_cfg, metrics_cfg, "validate")
        setup_metrics(self, cls_cfg, metrics_cfg, "test")

        self.classes = list(cls_cfg.keys())
        self.metric_names = [cfg.name.lower() for cfg in metrics_cfg.values()]

        # used for computing overall loss
        self.train_loss = []
        self.val_loss = []

    def forward(self, stage, batch):
        cap = batch['text'].long().cuda()
        mask = batch['mask'].cuda()
        target = batch['target'].cuda()
        mask_pos = batch['mask_pos'].cuda()

        logits = self.model(cap, mask, mask_pos)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        for cls_name in self.classes:
            indices = batch[f"{cls_name}_indices"]
            labels = batch[cls_name]
            logits = logits[indices]

            loss += F.cross_entropy(logits, labels)

            self.compute_metrics_step(stage, cls_name, labels, logits)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.forward("train", batch)
        self.log(f'train_loss', loss, sync_dist=True)
        self.train_loss.append(loss)

    def validation_step(self, batch, batch_idx):
        # this will be triggered during the Trainer's sanity check
        if not hasattr(self, "classes"):
            raise AttributeError("'classes' has not been initialised... Did you forget to call model.setup_tasks()?")
        
        loss = self.forward("validate", batch)
        self.log(f'validate_loss', loss, sync_dist=True)
        self.train_loss.append(loss)

    def test_step(self, batch, batch_idx):
        self.forward("test", batch)

    def predict_step(self, batch, batch_idx):
        output = self.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        results = {}
        for cls_name in self.classes:
            indices = batch[f"{cls_name}_indices"]
            logits = output.logits[indices]
            results[f"{cls_name}_preds"] = torch.argmax(logits, dim=1).tolist()

        return results
    
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

        opts = []
        for opt_cfg in self.optimizers:
            class_name = opt_cfg.pop("class_path")
            
            package_name = ".".join(class_name.split(".")[:-1])
            package = importlib.import_module(package_name)
            
            class_name = class_name.split(".")[-1]
            cls = getattr(package, class_name)

            opts.append(cls(optimizer_grouped_parameters, **opt_cfg))

        return opts