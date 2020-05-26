from typing import Dict, Any, List

import torch
from torch import Tensor
from torch.nn import Embedding, Linear, GRU

from torchrec.feature_column.CategoricalColumnWithIdentity import CategoricalColumnWithIdentity
from torchrec.model import IModel
from torchrec.utils.argument import ArgumentDescription


class GRU4Rec(IModel):
    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        pass

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        pass

    def __init__(self,
                 random_seed: int,
                 iid_column: CategoricalColumnWithIdentity,
                 his_len_column: CategoricalColumnWithIdentity,
                 his_column: CategoricalColumnWithIdentity,
                 label_column: CategoricalColumnWithIdentity,
                 emb_size: int,
                 hidden_size: int,
                 ):
        self.iid_column = iid_column
        self.his_len_column = his_len_column
        self.his_column = his_column
        self.label_column = label_column
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        super().__init__(random_seed)

    def _init_weights(self):
        self.i_embeddings = Embedding(self.iid_column.category_num, self.emb_size)
        self.rnn = GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
        self.out = Linear(self.hidden_size, self.emb_size, bias=False)

    def forward(self, data: Dict[str, Tensor]):
        i_ids: Tensor = self.iid_column.get_feature_data(data)  # [batch_size, sample_n]
        his_ids: Tensor = self.his_column.get_feature_data(data)  # [batch_size, max_his_len]
        his_len: Tensor = self.his_len_column.get_feature_data(data)  # [batch_size]

        i_vectors: Tensor = self.i_embeddings(i_ids)  # [batch_size, sample_n, emb_size]
        his_vectors: Tensor = self.i_embeddings(his_ids)  # [batch_size, max_his_len, emb_size]

        sort_his_lengths, sort_idx = torch.topk(his_len, k=len(his_len))  # noqa
        sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
        his_packed = torch.nn.utils.rnn.pack_padded_sequence(sort_his_vectors, sort_his_lengths, batch_first=True)

        # RNN
        output, hidden = self.rnn(his_packed, None)

        # Unsort
        sort_rnn_vector = self.out(hidden[-1])  # [batch_size, emb_size]
        unsort_idx = torch.topk(sort_idx, k=len(his_len), largest=False)[1]  # noqa
        rnn_vector: Tensor = sort_rnn_vector.index_select(dim=0, index=unsort_idx)  # [batch_size, emb_size]

        # Predicts
        prediction = (rnn_vector.unsqueeze(1) * i_vectors).sum(-1)  # [batch_size, sample_n]

        target = self.label_column.get_feature_data(data)
        if target is not None:
            target = target.float()

        return prediction, target
