from torch.optim import Adam

from torchrec.data.HistoryDataReader import HistoryDataReader
from torchrec.data.dataset.SplitMode import SplitMode
from torchrec.loss.BPRLoss import BPRLoss
from torchrec.metric.Hit import Hit
from torchrec.metric.NDCG import NDCG
from torchrec.model.GRU4Rec import GRU4Rec
from torchrec.task.Task import Task
from torchrec.task.TrainMode import TrainMode
from torchrec.utils.const import *

if __name__ == '__main__':
    data_reader = HistoryDataReader(
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
        max_his_len=10,
        use_neg_his=False,
    )

    feature_column_dict = data_reader.get_feature_column_dict()

    iid_column = feature_column_dict.get(IID)
    his_len_column = feature_column_dict.get(POS_HIS_LEN)
    his_column = feature_column_dict.get(POS_HIS)
    label_column = feature_column_dict.get(LABEL)

    model = GRU4Rec(
        random_seed=2020,
        iid_column=iid_column,
        his_len_column=his_len_column,
        his_column=his_column,
        label_column=label_column,
        emb_size=64,
        hidden_size=256,
    )

    optimizer = Adam(params=model.get_parameters(), lr=0.001, weight_decay=0.0)
    loss = BPRLoss()

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
        num_workers=6,
        dev_freq=1,
        filename="FunkSVD",
        monitor="ndcg@10",
        monitor_mode="max",
        patience=20,
        verbose=2,
    )

    print(task.run())
