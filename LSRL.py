from torch.nn import MSELoss
from torch.optim import Adam

from torchrec.data.ValueRLDataReader import ValueRLDataReader
from torchrec.data.dataset.SplitMode import SplitMode
from torchrec.metric.Hit import Hit
from torchrec.metric.NDCG import NDCG
from torchrec.model.LSRL import LSRLQNet, LSRL
from torchrec.task.Task import Task
from torchrec.task.TrainMode import TrainMode
from torchrec.utils.const import *

SEED = 2020

if __name__ == '__main__':
    data_reader = ValueRLDataReader(
        dataset="MovieLens-10M-PN",
        split_mode=SplitMode.LEAVE_K_OUT,
        warm_n=5,
        vt_ratio=0.1,
        leave_k=1,
        neg_sample_n=99,
        load_feature=False,
        append_id=False,
        train_mode=TrainMode.POINT_WISE,
        random_seed=SEED,
        max_state_len=10,
        use_neg_state=True,
        rl_sample_len=32,
    )

    feature_column_dict = data_reader.get_feature_column_dict()

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

    model = LSRL(
        random_seed=SEED,
        update_freq=1,
        gamma=0.9,
        reward_column=reward_column,
        q_net_type=LSRLQNet,
        weight_file="MovieLens-10M-PN.pt",
        uid_column=uid_column,
        iid_column=iid_column,
        pos_state_len_column=pos_state_len_column,
        pos_state_column=pos_state_column,
        pos_next_state_len_column=pos_next_state_len_column,
        pos_next_state_column=pos_next_state_column,
        neg_state_len_column=neg_state_len_column,
        neg_state_column=neg_state_column,
        neg_next_state_len_column=neg_next_state_len_column,
        neg_next_state_column=neg_next_state_column,
        rl_sample_column=rl_sample_column,
        emb_size=64,
        hidden_size=200,
    )

    optimizer = Adam(params=model.get_parameters(), lr=0.001, weight_decay=1e-6)
    loss = MSELoss()

    metrics = (
            [NDCG(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
            + [Hit(user_sample_n=100, k=k) for k in [1, 2, 5, 10, 20, 50, 100]]
    )

    task = Task(
        debug=False,
        gpu=0,
        random_seed=SEED,
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
        filename="FunkSVD",
        monitor="ndcg@10",
        monitor_mode="max",
        patience=50,
        verbose=2,
    )

    print(task.run())

# LSRL-v0 0.5467 0.5213
