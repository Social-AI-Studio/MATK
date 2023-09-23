import logging
import lightning.pytorch as pl

class BaseLightningModule(pl.LightningModule):
    def compute_metrics_step(self, cls_name, stage, loss, targets, preds):
        self.log(f'{stage}_{cls_name}_loss', loss, prog_bar=True)

        for metric_name in self.metric_names:
            metric = getattr(self, f"{stage}_{cls_name}_{metric_name}")
            metric(preds, targets)

    def compute_metrics_epoch(self, cls_name, stage):
        msg = "Epoch Results:\n"

        avg_metric_score = 0
        for metric_name in self.metric_names:
            metric = getattr(self, f"{stage}_{cls_name}_{metric_name}")
            metric_score = metric.compute()
            avg_metric_score += metric_score

            self.log(f'{stage}_{cls_name}_{metric_name}', metric_score,
                    prog_bar=True, sync_dist=True)
            
            msg += f"\t{stage}_{cls_name}_{metric_name}: {metric_score}\n"

            # reset the metrics
            metric.reset()

        avg_metric_score = avg_metric_score / len(self.metric_names)

        self.log(f'{stage}_{cls_name}_average', avg_metric_score,
                    prog_bar=True, sync_dist=True)
        
        msg += f"\t{stage}_{cls_name}_average: {avg_metric_score}\n"

        logging.info(msg)

    def on_training_epoch_end(self):
        for cls_name in self.classes:
            self.compute_metrics_epoch(cls_name, "train")

    def on_validation_epoch_end(self):
        for cls_name in self.classes:
            self.compute_metrics_epoch(cls_name, "validate")

    def on_test_epoch_end(self):
        for cls_name in self.classes:
            self.compute_metrics_epoch(cls_name, "test")