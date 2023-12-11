import logging
import lightning.pytorch as pl

class BaseLightningModule(pl.LightningModule):
    def compute_metrics_step(self, stage, cls_name, targets, preds):
        for metric_name in self.metric_names:
            metric = getattr(self, f"{stage}_{cls_name}_{metric_name}")
            if metric_name == 'rougescore':
                metric(preds, targets)['rougeL_fmeasure']
            else:
                metric(preds, targets)

    def compute_metrics_epoch(self, stage, cls_name):
        msg = "Epoch Results:\n"

        avg_metric_score = 0
        for metric_name in self.metric_names:
            metric = getattr(self, f"{stage}_{cls_name}_{metric_name}")
            metric_score = metric.compute()
            if metric_name == 'rougescore':
                metric_score = metric_score['rougeL_fmeasure']
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
            self.compute_metrics_epoch("train", cls_name)

    def on_validation_epoch_end(self):
        for cls_name in self.classes:
            self.compute_metrics_epoch("validate", cls_name)

    def on_test_epoch_end(self):
        for cls_name in self.classes:
            self.compute_metrics_epoch("test", cls_name)