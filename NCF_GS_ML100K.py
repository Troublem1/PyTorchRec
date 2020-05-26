from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from torchrec.data.SimpleDataReader import SimpleDataReader
from torchrec.data.dataset.SplitMode import SplitMode
from torchrec.metric.Hit import Hit
from torchrec.metric.NDCG import NDCG
from torchrec.model.NCF import NCF
from torchrec.task.GridSearch import GridSearch
from torchrec.task.GridSearch import create_params_list
from torchrec.task.TrainMode import TrainMode
from torchrec.utils.const import *

if __name__ == '__main__':
    data_reader_params = {
        "dataset": "MovieLens-100K-PN",
        "split_mode": SplitMode.LEAVE_K_OUT,
        "warm_n": 5,
        "vt_ratio": 0.1,
        "leave_k": 1,
        "neg_sample_n": 99,
        "load_feature": False,
        "append_id": False,
        "train_mode": TrainMode.PAIR_WISE,
        "random_seed": 2020,
    }

    feature_column_dict = SimpleDataReader(**data_reader_params).get_feature_column_dict()

    uid_column = feature_column_dict.get(UID)
    iid_column = feature_column_dict.get(IID)
    label_column = feature_column_dict.get(LABEL)

    model_params_list = create_params_list(
        base_params={
            "random_seed": 2020,
            "uid_column": uid_column,
            "iid_column": iid_column,
            "label_column": label_column,
            "emb_size": 64,
        },
        search_params={
            "layers": [[32], [64], [128]],
            "dropout": [0.2, 0.3, 0.4, 0.5, 0.6]
        },
    )

    optimizer_params_list = create_params_list(
        base_params={},
        search_params={
            "lr": [1e-3, 1e-4],
            "weight_decay": [0, 1e-6, 1e-5, 1e-4],
        }
    )

    loss = BCEWithLogitsLoss()

    metrics = (
            [NDCG(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
            + [Hit(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
    )

    task = GridSearch(
        gpu=0,
        random_seed=2020,
        metrics=metrics,
        train_mode=TrainMode.PAIR_WISE,
        data_reader_type=SimpleDataReader,
        data_reader_params=data_reader_params,
        model_type=NCF,
        model_params_list=model_params_list,
        epoch=100,
        batch_size=256,
        optimizer_type=Adam,
        optimizer_params_list=optimizer_params_list,
        loss=loss,
        num_workers=0,
        dev_freq=1,
        monitor="ndcg@10",
        monitor_mode="max",
        patience=20,
        verbose=2,
    )

    task.run()
