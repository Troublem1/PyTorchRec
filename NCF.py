from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from torchrec.data.SimpleDataReader import SimpleDataReader
from torchrec.data.dataset.SplitMode import SplitMode
from torchrec.metric.Hit import Hit
from torchrec.metric.NDCG import NDCG
from torchrec.model.NCF import NCF
from torchrec.task.Task import Task
from torchrec.task.TrainMode import TrainMode
from torchrec.utils.const import *

if __name__ == '__main__':
    data_reader = SimpleDataReader(
        dataset="MovieLens-100K-PN",
        split_mode=SplitMode.LEAVE_K_OUT,
        warm_n=5,
        vt_ratio=0.1,
        leave_k=1,
        neg_sample_n=99,
        load_feature=False,
        append_id=False,
        train_mode=TrainMode.PAIR_WISE,
        random_seed=2020,
    )

    feature_column_dict = data_reader.get_feature_column_dict()

    uid_column = feature_column_dict.get(UID)
    iid_column = feature_column_dict.get(IID)
    label_column = feature_column_dict.get(LABEL)

    model = NCF(
        random_seed=2020,
        uid_column=uid_column,
        iid_column=iid_column,
        label_column=label_column,
        emb_size=64,
        layers=[64],
        dropout=0.2,
    )

    optimizer = Adam(params=model.get_parameters(), lr=0.001, weight_decay=1e-4)
    loss = BCEWithLogitsLoss()

    metrics = (
            [NDCG(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
            + [Hit(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
    )

    task = Task(
        debug=False,
        gpu=0,
        random_seed=2020,
        metrics=metrics,
        train_mode=TrainMode.PAIR_WISE,
        data_reader=data_reader,
        model=model,
        epoch=100,
        batch_size=256,
        optimizer=optimizer,
        loss=loss,
        num_workers=2,
        dev_freq=1,
        filename="FunkSVD",
        monitor="ndcg@10",
        monitor_mode="max",
        patience=20,
        verbose=2,
    )

    print(task.run())
