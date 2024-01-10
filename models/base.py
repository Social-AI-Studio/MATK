import logging
import lightning.pytorch as pl

class BaseLightningModule(pl.LightningModule):
    def compute_metrics_step(self, stage, cls_name, targets, preds):
        for metric_name in self.metric_names:
            metric = getattr(self, f"{stage}_{cls_name}_{metric_name}")
            metric(preds, targets)

    def compute_metrics_epoch(self, stage):
        msg = f"{stage.capitalize()} Epoch Results:\n"

        total_avg_metric_score = 0
        for cls_name in self.classes:
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
            total_avg_metric_score += avg_metric_score

            self.log(f'{stage}_{cls_name}_average', avg_metric_score,
                     prog_bar=True, sync_dist=True)

            msg += f"\t{stage}_{cls_name}_average: {avg_metric_score}\n"

        final_avg_metric_score = total_avg_metric_score / len(self.classes)

        self.log(f'{stage}_average', final_avg_metric_score,
                 prog_bar=True, sync_dist=True)

        msg += f"\t{stage}_average' {final_avg_metric_score}\n"

        logging.info(msg)

    def on_training_epoch_end(self):
        self.compute_metrics_epoch("train")

    def on_validation_epoch_end(self):
        self.compute_metrics_epoch("validate")

    def on_test_epoch_end(self):
        self.compute_metrics_epoch("test")