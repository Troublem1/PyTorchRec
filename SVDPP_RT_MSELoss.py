from torch.nn import MSELoss
from torch.optim import Adam

from torchrec.data.SVDPPDataReader import SVDPPDataReader
from torchrec.data.dataset.SplitMode import SplitMode
from torchrec.metric.Hit import Hit
from torchrec.metric.NDCG import NDCG
from torchrec.model.SVDPP import SVDPP
from torchrec.task.RepeatTask import RepeatTask
from torchrec.task.TrainMode import TrainMode
from torchrec.utils.const import *


def repeat_task_on_dataset(dataset_name: str):
    data_reader_params = {
        "dataset": dataset_name,
        "split_mode": SplitMode.LEAVE_K_OUT,
        "warm_n": 5,
        "vt_ratio": 0.1,
        "leave_k": 1,
        "neg_sample_n": 99,
        "load_feature": False,
        "append_id": False,
        "train_mode": TrainMode.POINT_WISE,
        "random_seed": 2020,
        "limit": 256,
    }

    feature_column_dict = SVDPPDataReader(**data_reader_params).get_feature_column_dict()

    uid_column = feature_column_dict.get(UID)
    iid_column = feature_column_dict.get(IID)
    iids_column = feature_column_dict.get(IIDS)
    label_column = feature_column_dict.get(LABEL)

    model_params = {
        "random_seed": 2020,
        "uid_column": uid_column,
        "iid_column": iid_column,
        "iids_column": iids_column,
        "label_column": label_column,
        "emb_size": 64,
    }

    optimizer_params = {
        "lr": 1e-4,
        "weight_decay": 1e-5,
    }

    loss = MSELoss()

    metrics = (
            [NDCG(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
            + [Hit(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
    )

    task = RepeatTask(
        repeat_num=1,
        gpu=1,
        random_seed=2020,
        metrics=metrics,
        train_mode=TrainMode.POINT_WISE,
        data_reader_type=SVDPPDataReader,
        data_reader_params=data_reader_params,
        model_type=SVDPP,
        model_params=model_params,
        epoch=100,
        batch_size=256,
        optimizer_type=Adam,
        optimizer_params=optimizer_params,
        loss=loss,
        num_workers=5,
        dev_freq=1,
        monitor="ndcg@10",
        monitor_mode="max",
        patience=20,
        verbose=2,
    )

    task.run()


if __name__ == '__main__':
    # repeat_task_on_dataset("MovieLens-100K-PN")
    repeat_task_on_dataset("MovieLens-10M-PN")
    repeat_task_on_dataset("MovieLens-1M-PN")
