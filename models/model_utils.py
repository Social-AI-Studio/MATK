import torch.nn as nn
import torchmetrics

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

def setup_metrics(obj, cls_dict, metrics_cfg, stage):
    for cls_name, num_classes in cls_dict.items():
        for cfg in metrics_cfg.values():
            metric_name = cfg.pop("name")
            metric_class = getattr(torchmetrics, metric_name)
            # e.g., train_validate_auroc
            setattr(
                obj,
                f"{stage}_{cls_name}_{metric_name.lower()}",
                metric_class(num_classes=num_classes, **cfg)
            )
            cfg["name"] = metric_name