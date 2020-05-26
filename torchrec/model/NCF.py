from typing import Dict, Any, List

import torch
from torch import Tensor
from torch.nn import Embedding, Linear

from torchrec.feature_column.CategoricalColumnWithIdentity import CategoricalColumnWithIdentity
from torchrec.model import IModel
from torchrec.model.layer.MLP import MLP
from torchrec.utils.argument import ArgumentDescription


class NCF(IModel):
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
                 label_column: CategoricalColumnWithIdentity,
                 emb_size: int,
                 layers: List[int],
                 dropout: float):
        self.uid_column = uid_column
        self.iid_column = iid_column
        self.label_column = label_column
        self.emb_size = emb_size
        self.layers = layers
        self.dropout = dropout
        super().__init__(random_seed)

    def _init_weights(self):
        self.mf_u_embeddings = Embedding(self.uid_column.category_num, self.emb_size)
        self.mf_i_embeddings = Embedding(self.iid_column.category_num, self.emb_size)
        self.mlp_u_embeddings = Embedding(self.uid_column.category_num, self.emb_size)
        self.mlp_i_embeddings = Embedding(self.iid_column.category_num, self.emb_size)

        self.mlp = MLP(
            input_units=2 * self.emb_size,
            hidden_units_list=self.layers,
            activation="relu",
            dropout=self.dropout,
        )

        self.prediction = Linear(self.emb_size + self.layers[-1], 1, bias=False)

    def forward(self, data: Dict[str, Tensor]):
        u_ids: Tensor = self.uid_column.get_feature_data(data)  # [batch_size]
        i_ids: Tensor = self.iid_column.get_feature_data(data)  # [batch_size, -1]

        sample_n = i_ids.shape[1]

        u_ids: Tensor = u_ids.unsqueeze(-1).repeat(1, sample_n).reshape(-1)  # [batch_size * -1]
        i_ids: Tensor = i_ids.reshape(-1)  # [batch_size * -1]

        mf_u_vectors: Tensor = self.mf_u_embeddings(u_ids)  # [batch_size * -1, emb_size]
        mf_i_vectors: Tensor = self.mf_i_embeddings(i_ids)  # [batch_size * -1, emb_size]
        mlp_u_vectors: Tensor = self.mlp_u_embeddings(u_ids)  # [batch_size * -1, emb_size]
        mlp_i_vectors: Tensor = self.mlp_i_embeddings(i_ids)  # [batch_size * -1, emb_size]

        mf_vector: Tensor = mf_u_vectors * mf_i_vectors  # [batch_size * -1, emb_size]
        mlp_vector: Tensor = torch.cat([mlp_u_vectors, mlp_i_vectors], dim=-1)  # [batch_size * -1, 2 * emb_size]

        mlp_vector: Tensor = self.mlp(mlp_vector)  # [batch_size * -1, last_dense_size]

        # [batch_size * -1, last_dense_size + emb_size]
        output_vector: Tensor = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction: Tensor = self.prediction(output_vector).reshape(-1, sample_n)  # [batch_size, -1]

        target = torch.zeros_like(prediction, dtype=torch.float32)
        target[:, 0] = 1

        return prediction, target
