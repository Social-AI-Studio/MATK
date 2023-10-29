import torch, importlib
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification

from .model_utils import setup_metrics
from .base import BaseLightningModule


class RobertaClassificationModel(BaseLightningModule):
    def __init__(
            self,
            model_class_or_path: str,
            num_classes: int
    ):
        super().__init__()
        self.save_hyperparameters()

        # set up base model
        self.base_model = RobertaForSequenceClassification.from_pretrained(
            model_class_or_path,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=True
        )

        self.train_loss = []
        self.val_loss = []

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
        output = self.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        loss = 0
        for cls_name in self.classes:
            indices = batch[f"{cls_name}_indices"]
            labels = batch[cls_name]
            logits = output.logits[indices]

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
        opts = []
        for opt_cfg in self.optimizers:
            class_name = opt_cfg.pop("class_path")
            
            package_name = ".".join(class_name.split(".")[:-1])
            package = importlib.import_module(package_name)
            
            class_name = class_name.split(".")[-1]
            cls = getattr(package, class_name)

            opts.append(cls(self.parameters(), **opt_cfg))

        return opts