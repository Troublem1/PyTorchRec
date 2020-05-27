from typing import Dict, Any, List

import torch
from torch import Tensor
from torch.nn import Embedding, Parameter

from torchrec.feature_column.CategoricalColumnWithIdentity import CategoricalColumnWithIdentity
from torchrec.model import IModel
from torchrec.utils.argument import ArgumentDescription


class SVDPP(IModel):
    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        pass

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        pass

    def __init__(self,
                 random_seed: int,
                 uid_column: CategoricalColumnWithIdentity,
                 iid_column: CategoricalColumnWithIdentity,
                 iids_column: CategoricalColumnWithIdentity,
                 label_column: CategoricalColumnWithIdentity,
                 emb_size: int,
                 ):
        self.uid_column = uid_column
        self.iid_column = iid_column
        self.iids_column = iids_column
        self.label_column = label_column
        self.emb_size = emb_size
        super().__init__(random_seed)

    def _init_weights(self):
        self.u_embeddings = Embedding(self.uid_column.category_num, self.emb_size)
        self.i_embeddings = Embedding(self.iid_column.category_num, self.emb_size)
        self.implicit_i_embeddings = Embedding(self.iid_column.category_num, self.emb_size)
        self.u_bias = Embedding(self.uid_column.category_num, 1)
        self.i_bias = Embedding(self.iid_column.category_num, 1)
        self.global_bias = Parameter(torch.tensor(0.0))

    def forward(self, data: Dict[str, Tensor]):
        u_ids: Tensor = self.uid_column.get_feature_data(data)  # [batch_size]
        i_ids: Tensor = self.iid_column.get_feature_data(data)  # [batch_size] or [batch_size, sample_n]
        implicit_i_ids: Tensor = self.iids_column.get_feature_data(data)  # [batch_size, his_len]

        valid_implicit_i_ids: Tensor = implicit_i_ids.gt(0).float()  # [batch_size, his_len]
        implicit_i_vector: Tensor = self.implicit_i_embeddings(implicit_i_ids)  # [batch_size, his_len, emb_size]
        # [batch_size, emb_size]
        implicit_i_vector: Tensor = (implicit_i_vector * valid_implicit_i_ids.unsqueeze(dim=-1)).sum(dim=1)
        valid_implicit_i_ids_len: Tensor = valid_implicit_i_ids.sum(dim=-1)  # [batch_size]
        # [batch_size, emb_size]
        implicit_i_vector: Tensor = implicit_i_vector / valid_implicit_i_ids_len.sqrt().unsqueeze(dim=-1)

        u_vectors: Tensor = self.u_embeddings(u_ids)  # [batch_size, emb_size]
        i_vectors: Tensor = self.i_embeddings(i_ids)  # [batch_size, emb_size] or [batch_size, sample_n, emb_size]

        u_bias: Tensor = self.u_bias(u_ids).squeeze(1)  # [batch_size]
        i_bias: Tensor = self.i_bias(i_ids).squeeze(1)  # [batch_size] or [batch_size, sample_n]

        if len(i_ids.shape) == 1:
            # [batch_size]
            prediction: Tensor = (((u_vectors + implicit_i_vector) * i_vectors).sum(dim=-1)
                                  + u_bias + i_bias + self.global_bias)

            target = self.label_column.get_feature_data(data)
            if target is not None:
                target = target.float()

        else:
            sample_n: int = i_ids.shape[1]
            # [batch_size * sample_n, emb_size]
            u_vectors: Tensor = u_vectors.unsqueeze(1).repeat(1, sample_n, 1).reshape(-1, self.emb_size)
            # [batch_size * sample_n, emb_size]
            implicit_i_vector: Tensor = implicit_i_vector.unsqueeze(1).repeat(1, sample_n, 1).reshape(-1, self.emb_size)
            # [batch_size * sample_n, emb_size]
            i_vectors: Tensor = i_vectors.reshape(-1, self.emb_size)
            # [batch_size * sample_n]
            u_bias: Tensor = u_bias.unsqueeze(1).repeat(1, sample_n).reshape(-1)
            # [batch_size * sample_n]
            i_bias: Tensor = i_bias.reshape(-1)
            # [batch_size, sample_n]
            prediction: Tensor = (((u_vectors + implicit_i_vector) * i_vectors).sum(dim=-1)
                                  + u_bias + i_bias + self.global_bias).reshape(-1, sample_n)

            target = torch.zeros_like(prediction, dtype=torch.float32)
            target[:, 0] = 1

        return prediction, target
