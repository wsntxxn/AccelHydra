from dataclasses import dataclass
from pathlib import Path
import datetime

from accel_hydra import Trainer


@dataclass(kw_only=True)
class MnistTrainer(Trainer):
    def training_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.model(features)
        loss = self.loss_fn(preds, labels)
        lr = self.optimizer.param_groups[0]["lr"]
        self.accelerator.log({"train/lr": lr}, step=self.step)
        return loss

    def on_validation_start(self):
        self.validation_stats = {"accurate": 0, "num_elems": 0}

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        preds = self.model(features)
        predictions = preds.argmax(dim=-1)
        output = {"predictions": predictions, "labels": labels}
        output = self.accelerator.gather_for_metrics(output)
        accurate_preds = (output["predictions"] == output["labels"])
        self.validation_stats["accurate"] += accurate_preds.long().sum()
        self.validation_stats["num_elems"] += accurate_preds.shape[0]

    def get_val_metrics(self):
        return {
            "accuracy":
                self.validation_stats["accurate"].item() /
                self.validation_stats["num_elems"]
        }

    def on_validation_end(self):
        eval_metric = self.validation_stats["accurate"].item(
        ) / self.validation_stats["num_elems"]
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.accelerator.print(
            f"epoch[{self.epoch}]@{nowtime} --> eval_metric= {100 * eval_metric:.2f}%"
        )
        self.accelerator.log({"val/accuracy": eval_metric}, step=self.step)
