import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from .model_utils import setup_metrics, SimpleClassifier
from .base import BaseLightningModule

import torch
import torch.nn as nn
import torch.nn.functional as F

            
# def sche_lambda(step):
#     warm_up_steps = 95 # harmeme 
#     return min(step/warm_up_steps, 1) 

class GateFusionModule(nn.Module):
    def __init__(self, vla_dims, mie_dims, output_embeds):
        super(GateFusionModule, self).__init__()
        # Transform layers to common space
        self.vla_transform = nn.Linear(vla_dims, output_embeds)
        self.mie_transform = nn.Linear(mie_dims, output_embeds)
        
        # Gate layer
        self.gate_layer = nn.Linear(2 * output_embeds, output_embeds)
        self.dropout = nn.Dropout(0.1)

        # Transform layers for residual connections
        self.vla_residual = nn.Linear(vla_dims, output_embeds)
        self.mie_residual = nn.Linear(mie_dims, output_embeds)
    
    def forward(self, vla_embeds, mie_embeds, add_residual_conn):
        # Transform inputs to a common space
        vla_common = self.dropout(F.relu(self.vla_transform(vla_embeds)))
        mie_common = self.dropout(F.relu(self.mie_transform(mie_embeds)))

        # Concatenate vla_embeds and input2 along the feature dimension
        combined_inputs = torch.cat((vla_common, mie_common), dim=1)
        
        # Compute the gate based on combined inputs
        gate = torch.sigmoid(self.gate_layer(combined_inputs))
        fused_output = gate * vla_common + (1 - gate) * mie_common

        # Add residual connections
        if add_residual_conn:
            fused_output += self.vla_residual(vla_embeds)
            fused_output += self.mie_residual(mie_embeds)

        return fused_output

class IntMemeModel(BaseLightningModule):
    def __init__(
        self,
        vla_class_or_path: str,
        mie_class_or_path: str,
        dropout: float,
        optimizers: list,
        output_dims: int,
        add_residual_conn: bool,
        classes_per_task: list
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vla_model = AutoModel.from_pretrained(vla_class_or_path)
        self.mie_model = AutoModel.from_pretrained(mie_class_or_path)

        self.dropout = dropout
        self.optimizers = optimizers
        self.add_residual_conn = add_residual_conn

        # set up the gate attention layer
        self.gate = GateFusionModule(
            self.vla_model.config.multimodal_config.hidden_size,
            self.mie_model.config.hidden_size,
            output_dims
        )

        # set up the various classification heads
        self.mlps = nn.ModuleList([
            SimpleClassifier(
                output_dims,
                num_classes,
                self.dropout
            )
            for num_classes in classes_per_task
        ])

    def setup_tasks(self, metrics_cfg, cls_cfg):
        # set up the metrics for evaluation
        setup_metrics(self, cls_cfg, metrics_cfg, "train")
        setup_metrics(self, cls_cfg, metrics_cfg, "validate")
        setup_metrics(self, cls_cfg, metrics_cfg, "test")

        # important variables used in the BaseLightningModule
        self.classes = list(cls_cfg.keys())
        self.metric_names = [cfg["name"].lower() for cfg in metrics_cfg.values()]

        # used for computing overall loss
        self.train_loss = []
        self.val_loss = []

    def forward(self, stage, batch):
        vla_outputs = self.vla_model(
            input_ids=batch["meme_input_ids"],
            attention_mask=batch["meme_attention_mask"],
            pixel_values=batch['pixel_values']
        )
        vla_cls = vla_outputs.multimodal_embeddings[:, 0]

        mie_outputs = self.mie_model(
            input_ids=batch["passage_input_ids"],
            attention_mask=batch["passage_attention_mask"]
        )
        mie_cls = mie_outputs.last_hidden_state[:, 0]
        fused_outputs = self.gate(vla_cls, mie_cls, self.add_residual_conn)

        loss = 0.0
        for idx, cls_name in enumerate(self.classes):
            indices = batch[f"{cls_name}_indices"]
            targets = batch[cls_name]


            classifier = self.mlps[idx]
            logits = classifier(fused_outputs)

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
            raise AttributeError(
                "'classes' has not been initialised... Did you forget to call model.setup_tasks()?")

        loss = self.forward("validate", batch)
        self.val_loss.append(loss)

        self.log(f'validate_loss', loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        self.forward("test", batch)
        return

    def predict_step(self, batch):
        vla_outputs = self.vla_model(
            input_ids=batch["meme_input_ids"],
            attention_mask=batch["meme_attention_mask"],
            pixel_values=batch['pixel_values']
        )
        vla_cls = vla_outputs.multimodal_embeddings[:, 0]

        mie_outputs = self.mie_model(
            input_ids=batch["passage_input_ids"],
            attention_mask=batch["passage_attention_mask"]
        )
        mie_cls = mie_outputs.last_hidden_state[:, 0]
        fused_outputs = self.gate(vla_cls, mie_cls, self.add_residual_conn)

        for idx, cls_name in enumerate(self.classes):
            classifier = self.mlps[idx]
            logits = classifier(fused_outputs)

        results = {
            "logits": [],
        }
        for idx, cls_name in enumerate(self.classes):
            results["logits"] = logits.detach().cpu().tolist()

        return results
    

    # def configure_optimizers(self):
    #     opts = []
    #     for opt_cfg in self.optimizers:
    #         class_name = opt_cfg.pop("class_path")

    #         package_name = ".".join(class_name.split(".")[:-1])
    #         package = importlib.import_module(package_name)

    #         class_name = class_name.split(".")[-1]
    #         cls = getattr(package, class_name)

    #         opts.append(cls(self.parameters(), **opt_cfg))

    #     return opts

    def configure_optimizers(self):
        optimizers = []
        for opt_cfg in self.optimizers:
            class_name = opt_cfg.pop("class_path")

            package_name = ".".join(class_name.split(".")[:-1])
            package = importlib.import_module(package_name)

            class_name = class_name.split(".")[-1]
            cls = getattr(package, class_name)

            opt = cls(self.parameters(), **opt_cfg)

            optimizers.append(opt)
            # schedulers.append(torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=sche_lambda))

        return optimizers #, schedulers
