from typing import Dict, Any, List, Iterable, Tuple

import pandas as pd
from numpy.random._generator import default_rng  # noqa
from pandas import DataFrame
from torch.nn.modules.loss import _Loss  # noqa
from torch.optim import Optimizer

from torchrec.data.IDataReader import IDataReader
from torchrec.metric import IMetric
from torchrec.model import IModel
from torchrec.task import TrainMode
from torchrec.task.ITask import ITask
from torchrec.task.Task import Task
from torchrec.utils.argument import ArgumentDescription


class GridSearch(ITask):
    def __init__(self,
                 gpu: int,
                 random_seed: int,
                 metrics: List[IMetric],
                 train_mode: TrainMode,
                 data_readers: Iterable[Tuple[IDataReader, Dict]],
                 models: Iterable[Tuple[IModel, Dict]],
                 epoch: int,
                 batch_size: int,
                 optimizers: Iterable[Tuple[Optimizer, Dict]],
                 loss: _Loss,
                 num_workers: int,
                 dev_freq: int,
                 monitor: str,
                 monitor_mode: str,
                 patience: int,
                 ):
        self.gpu = gpu
        self.random_seed = random_seed
        self.metrics: List[IMetric] = metrics
        self.train_mode = train_mode
        self.data_readers = data_readers
        self.models = models
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizers = optimizers
        self.loss = loss
        self.num_workers = num_workers
        self.dev_freq = dev_freq
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.patience = patience

    def run(self):
        task_logs: Dict[str, List] = {}
        dataset_name = None
        model_name = None
        for data_reader, data_params in self.data_readers:
            for model, model_params in self.models:
                for optimizer, optimizer_params in self.optimizers:
                    dataset_name = data_reader.dataset
                    model_name = model.__class__.__name__
                    params: Dict[str, str] = {
                        "model": model.__class__.__name__,
                        "dataset": data_reader.dataset,
                        "seed": str(self.random_seed),
                        "train": self.train_mode.value,
                        "epoch": str(self.epoch),
                        "batch_size": str(self.batch_size),
                        "loss": self.loss.__class__.__name__,
                        "dev_freq": str(self.dev_freq),
                        "monitor": self.monitor,
                        "patience": str(self.patience),
                    }
                    params.update(data_params)
                    params.update(model_params)
                    params.update(optimizer_params)
                    filename: str = "-".join([param[:3] + params[param] for param in params])
                    best_epoch, best_dev_logs, test_logs = Task(
                        gpu=self.gpu,
                        random_seed=self.random_seed,
                        metrics=self.metrics,
                        train_mode=self.train_mode,
                        data_reader=data_reader,
                        model=model,
                        epoch=self.epoch,
                        batch_size=self.batch_size,
                        optimizer=optimizer,
                        loss=self.loss,
                        num_workers=self.num_workers,
                        dev_freq=self.dev_freq,
                        filename=filename,
                        monitor=self.monitor,
                        monitor_mode=self.monitor_mode,
                        patience=self.patience,
                    ).run()
                    params["best_epoch"] = best_epoch
                    params.update({f"dev_{key}": best_dev_logs[key] for key in best_dev_logs})
                    params.update({f"test_{key}": test_logs[key] for key in test_logs})
                    for key in params:
                        task_logs.setdefault(key, []).append(params[key])
        grid_search_df: DataFrame = pd.DataFrame(task_logs)
        grid_search_df.to_csv(
            path_or_buf=f"{model_name}_{dataset_name}_grid_search.csv",
            sep="\t",
        )

    @classmethod
    def create_from_console(cls):
        pass

    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        pass

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        pass
