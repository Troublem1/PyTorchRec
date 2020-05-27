from typing import Dict, Any, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Embedding
from torch.nn.modules.loss import _Loss  # noqa

from torchrec.feature_column.CategoricalColumnWithIdentity import CategoricalColumnWithIdentity
from torchrec.model import IModel
from torchrec.model.utils import get_valid_his_index, get_postion_ids
from torchrec.utils.argument import ArgumentDescription


def scaled_dot_product_attention(q, k, v, scale=None, attn_mask=None):
    """
    Weighted sum of v according to dot product between q and k
    :param q: query tensor，[-1, L_q, V]
    :param k: key tensor，[-1, L_k, V]
    :param v: value tensor，[-1, L_k, V]
    :param scale: scalar
    :param attn_mask: [-1, L_q, L_k]
    """
    attention = torch.bmm(q, k.transpose(1, 2))  # [-1, L_q, L_k]
    if scale is not None:
        attention = attention * scale
    attention = attention - attention.max()
    if attn_mask is not None:
        attention = attention.masked_fill(attn_mask, -np.inf)
    attention = attention.softmax(dim=-1)
    context = torch.bmm(attention, v)  # [-1, L_q, V]
    return context


class SASRec(IModel):
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
                 max_his_len: int,
                 num_layers: int,
                 dropout: float,
                 ):
        self.iid_column = iid_column
        self.his_len_column = his_len_column
        self.his_column = his_column
        self.label_column = label_column
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.max_his_len = max_his_len
        self.num_layers = num_layers
        self.dropout = dropout
        super().__init__(random_seed)

    def _init_weights(self):
        self.i_embeddings = Embedding(self.iid_column.category_num, self.emb_size)
        self.p_embeddings = Embedding(self.max_his_len + 1, self.emb_size)

        self.Q = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.K = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.W1 = torch.nn.Linear(self.emb_size, self.emb_size)
        self.W2 = torch.nn.Linear(self.emb_size, self.emb_size)

        self.dropout_layer = torch.nn.Dropout(p=self.dropout)
        self.layer_norm = torch.nn.LayerNorm(self.emb_size)

    def forward(self, data: Dict[str, Tensor]):
        i_ids: Tensor = self.iid_column.get_feature_data(data)  # [batch_size, sample_n]
        his_ids: Tensor = self.his_column.get_feature_data(data)  # [batch_size, max_his_len]
        his_len: Tensor = self.his_len_column.get_feature_data(data)  # [batch_size]

        valid_his_index = get_valid_his_index(his_ids)  # [batch_size, max_his_len]

        i_vectors: Tensor = self.i_embeddings(i_ids)  # [batch_size, sample_n, emb_size]
        his_vectors: Tensor = self.i_embeddings(his_ids)  # [batch_size, max_his_len, emb_size]

        # position embedding
        # lengths:  [4, 2, 5]
        # position: [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0], [5, 4, 3, 2, 1]]
        pos_ids = get_postion_ids(valid_his_index, his_len, self.compiled_device)  # [batch_size, max_his_len]
        pos_vectors = self.p_embeddings(pos_ids)
        his_vectors = his_vectors + pos_vectors

        attention_mask: Tensor = -valid_his_index.unsqueeze(1).repeat(1, self.max_his_len, 1) + 1
        for i in range(self.num_layers):
            residual = his_vectors
            query, key = self.Q(his_vectors), self.K(his_vectors)  # [batch_size, history_max, emb_size]
            # self-attention
            scale = self.emb_size ** -0.5
            context = scaled_dot_product_attention(query, key, key, scale=scale, attn_mask=attention_mask)
            # mlp forward
            context = self.W1(context).relu()
            his_vectors = self.W2(context)  # [batch_size, history_max, emb_size]
            # dropout, residual and layer_norm
            his_vectors = self.dropout_layer(his_vectors)
            his_vectors = self.layer_norm(residual + his_vectors)

        his_vector = (his_vectors * valid_his_index.unsqueeze(-1).float()).sum(1)  # [batch_size, emb_size]
        his_vector = his_vector / his_len.unsqueeze(-1).float()

        prediction = (his_vector.unsqueeze(1) * i_vectors).sum(-1)

        target = self.label_column.get_feature_data(data)
        if target is not None:
            target = target.float()

        return prediction, target
