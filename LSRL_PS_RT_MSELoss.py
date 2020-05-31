from torch.nn import MSELoss
from torch.optim import Adam

from torchrec.data.ValueRLDataReader import ValueRLDataReader
from torchrec.data.dataset.SplitMode import SplitMode
from torchrec.metric.Hit import Hit
from torchrec.metric.NDCG import NDCG
from torchrec.model.LSRL_PS import LSRLPSQNet, LSRLPS
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
        "max_state_len": 10,
        "use_neg_state": True,
        "rl_sample_len": 32,
    }

    feature_column_dict = ValueRLDataReader(**data_reader_params).get_feature_column_dict()

    uid_column = feature_column_dict.get(UID)
    iid_column = feature_column_dict.get(IID)
    pos_state_len_column = feature_column_dict.get(POS_STATE_LEN)
    pos_state_column = feature_column_dict.get(POS_STATE)
    pos_next_state_len_column = feature_column_dict.get(POS_NEXT_STATE_LEN)
    pos_next_state_column = feature_column_dict.get(POS_NEXT_STATE)
    neg_state_len_column = feature_column_dict.get(NEG_STATE_LEN)
    neg_state_column = feature_column_dict.get(NEG_STATE)
    neg_next_state_len_column = feature_column_dict.get(NEG_NEXT_STATE_LEN)
    neg_next_state_column = feature_column_dict.get(NEG_NEXT_STATE)
    rl_sample_column = feature_column_dict.get(RL_SAMPLE)
    reward_column = feature_column_dict.get(LABEL)

    model_params = {
        "random_seed": 2020,
        "reward_column": reward_column,
        "q_net_type": LSRLPSQNet,
        "weight_file": "MovieLens-100K-PN.pt",
        "uid_column": uid_column,
        "iid_column": iid_column,
        "pos_state_len_column": pos_state_len_column,
        "pos_state_column": pos_state_column,
        "pos_next_state_len_column": pos_next_state_len_column,
        "pos_next_state_column": pos_next_state_column,
        "neg_state_len_column": neg_state_len_column,
        "neg_state_column": neg_state_column,
        "neg_next_state_len_column": neg_next_state_len_column,
        "neg_next_state_column": neg_next_state_column,
        "rl_sample_column": rl_sample_column,
        "emb_size": 64,
        "hidden_size": 200,
        "update_freq": 1,
        "gamma": 0.9
    }

    optimizer_params = {
        "lr": 1e-3,
        "weight_decay": 0.0,
    }

    loss = MSELoss()

    metrics = (
            [NDCG(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
            + [Hit(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
    )

    task = RepeatTask(
        repeat_num=1,
        gpu=7,
        random_seed=2020,
        metrics=metrics,
        train_mode=TrainMode.POINT_WISE,
        data_reader_type=ValueRLDataReader,
        data_reader_params=data_reader_params,
        model_type=LSRLPS,
        model_params=model_params,
        epoch=100,
        batch_size=256,
        optimizer_type=Adam,
        optimizer_params=optimizer_params,
        loss=loss,
        num_workers=1,
        dev_freq=1,
        monitor="ndcg@10",
        monitor_mode="max",
        patience=20,
        verbose=2,
    )

    task.run()


if __name__ == '__main__':
    repeat_task_on_dataset("MovieLens-100K-PN")
    repeat_task_on_dataset("MovieLens-1M-PN")
    repeat_task_on_dataset("MovieLens-10M-PN")
