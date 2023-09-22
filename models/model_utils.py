import torchmetrics


def setup_metrics(obj, cls_dict, metrics_cfg, stage):
    for cls_name, num_classes in cls_dict.items():
        for cfg in metrics_cfg.values():
            metric_name = cfg.name
            metric_class = getattr(torchmetrics, metric_name)

            # e.g., train_validate_auroc
            setattr(
                obj,
                f"{stage}_{cls_name}_{metric_name.lower()}",
                metric_class(num_classes=num_classes, **cfg)
            )

def collapse_cls_dict(cls_dict):
    collapsed_dict = {}
    for dataset, label_dict in cls_dict.items():
        for label, value in label_dict.items():
            collapsed_key = f"{dataset}_{label}"
            collapsed_dict[collapsed_key] = value
    return collapsed_dict

