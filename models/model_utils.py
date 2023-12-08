import torch.nn as nn
import torchmetrics
import pkgutil

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(SimpleClassifier,self).__init__()
        layer=[
            nn.Linear(in_dim,out_dim),
            nn.Dropout(dropout,inplace=True)
        ]
        self.main = nn.Sequential(*layer)
        
    def forward(self,x):
        logits = self.main(x)
        return logits    

def find_metric_class(submodule, metric_name):

    # Iterate over all submodules in torchmetrics
    for _, module_name, _ in pkgutil.iter_modules(submodule.__path__):
        try:
            current_submodule = getattr(submodule, module_name)
        except AttributeError:
            continue

        # Try to find the metric class in the current submodule
        try:
            metric_class = getattr(current_submodule, metric_name)
            return metric_class
        except AttributeError:
            pass

        # Recursively search for the metric in submodules
        try:
            metric_class = find_metric_class(metric_name, current_submodule)
            if metric_class is not None:
                return metric_class
        except Exception:
            pass

    raise ValueError(f"Metric '{metric_name}' not found in torchmetrics.")

def setup_metrics(obj, cls_dict, metrics_cfg, stage):
    for cls_name, num_classes in cls_dict.items():
        for cfg in metrics_cfg.values():
            metric_name = cfg.pop("name")
            metric_class = find_metric_class(torchmetrics, metric_name)
            # e.g., train_validate_auroc
            setattr(
                obj,
                f"{stage}_{cls_name}_{metric_name.lower()}",
                metric_class(num_classes=num_classes, **cfg)
            )
            cfg["runtime_name"] = metric_name

def setup_generation_metrics(obj, metrics_cfg, stage):
    for cfg in metrics_cfg.values():
        metric_name = cfg.pop("name")
        if stage != "train":
            runtime_name = cfg.pop("runtime_name")
        metric_class = find_metric_class(torchmetrics, metric_name)
        # e.g., train_validate_auroc

        if metric_name == "BLEUScore" and "weights" in cfg.keys() and cfg["weights"]==[1.0,0.0,0.0,0.0]:
            setattr(
                obj,
                f"{stage}_target_bleu1",
                metric_class(**cfg)
            )
            cfg["runtime_name"] = 'bleu1'
        elif metric_name == "BLEUScore" and "weights" in cfg.keys() and cfg["weights"]==[0.0,1.0,0.0,0.0]:
            setattr(
                obj,
                f"{stage}_target_bleu2",
                metric_class(**cfg)
            )
            cfg["runtime_name"] = 'bleu2'
        else:
            setattr(
                obj,
                f"{stage}_target_{metric_name.lower()}",
                metric_class(**cfg)
            )
            cfg["runtime_name"] = metric_name
        cfg["name"] = metric_name

        