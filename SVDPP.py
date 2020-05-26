from torch.nn import MSELoss
from torch.optim import Adam

from torchrec.data.SVDPPDataReader import SVDPPDataReader
from torchrec.data.dataset.SplitMode import SplitMode
from torchrec.metric.Hit import Hit
from torchrec.metric.NDCG import NDCG
from torchrec.model.SVDPP import SVDPP
from torchrec.task.Task import Task
from torchrec.task.TrainMode import TrainMode
from torchrec.utils.const import *

if __name__ == '__main__':
    data_reader = SVDPPDataReader(
        dataset="MovieLens-100K-PN",
        split_mode=SplitMode.LEAVE_K_OUT,
        warm_n=5,
        vt_ratio=0.1,
        leave_k=1,
        neg_sample_n=99,
        load_feature=False,
        append_id=False,
        train_mode=TrainMode.POINT_WISE,
        random_seed=2020,
        limit=256,
    )

    feature_column_dict = data_reader.get_feature_column_dict()

    uid_column = feature_column_dict.get(UID)
    iid_column = feature_column_dict.get(IID)
    iids_column = feature_column_dict.get(IIDS)
    label_column = feature_column_dict.get(LABEL)

    model = SVDPP(
        random_seed=2020,
        uid_column=uid_column,
        iid_column=iid_column,
        iids_column=iids_column,
        label_column=label_column,
        emb_size=64,
    )

    optimizer = Adam(params=model.get_parameters(), lr=0.001, weight_decay=1e-4)
    loss = MSELoss()

    metrics = (
            [NDCG(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
            + [Hit(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
    )

    task = Task(
        debug=False,
        gpu=1,
        random_seed=2020,
        metrics=metrics,
        train_mode=TrainMode.POINT_WISE,
        data_reader=data_reader,
        model=model,
        epoch=100,
        batch_size=256,
        optimizer=optimizer,
        loss=loss,
        num_workers=6,
        dev_freq=1,
        filename="SVDPP",
        monitor="ndcg@10",
        monitor_mode="max",
        patience=20,
        verbose=2,
    )

    print(task.run())
