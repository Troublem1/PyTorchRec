from torch.nn import MSELoss
from torch.optim import Adam

from torchrec.data.ValueRLDataReader import ValueRLDataReader
from torchrec.data.dataset.SplitMode import SplitMode
from torchrec.metric.Hit import Hit
from torchrec.metric.NDCG import NDCG
from torchrec.model.DQN import DQN, DQNQNet
from torchrec.task.Task import Task
from torchrec.task.TrainMode import TrainMode
from torchrec.utils.const import *

if __name__ == '__main__':
    data_reader = ValueRLDataReader(
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
        max_state_len=10,
        use_neg_state=False,
        rl_sample_len=32,
    )

    feature_column_dict = data_reader.get_feature_column_dict()

    iid_column = feature_column_dict.get(IID)
    state_len_column = feature_column_dict.get(POS_STATE_LEN)
    state_column = feature_column_dict.get(POS_STATE)
    next_state_len_column = feature_column_dict.get(POS_NEXT_STATE_LEN)
    next_state_column = feature_column_dict.get(POS_NEXT_STATE)
    rl_sample_column = feature_column_dict.get(RL_SAMPLE)
    reward_column = feature_column_dict.get(LABEL)

    model = DQN(
        random_seed=2020,
        update_freq=2,
        gamma=0.9,
        reward_column=reward_column,
        q_net_type=DQNQNet,
        weight_file="MovieLens-100K-PN.pt",
        iid_column=iid_column,
        state_len_column=state_len_column,
        state_column=state_column,
        next_state_len_column=next_state_len_column,
        next_state_column=next_state_column,
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
        filename="FunkSVD",
        monitor="ndcg@10",
        monitor_mode="max",
        patience=50,
        verbose=2,
    )

    print(task.run())
