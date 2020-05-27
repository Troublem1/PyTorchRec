from typing import Dict, Any, List

import torch
from torch import Tensor
from torch.nn import Embedding

from torchrec.feature_column.CategoricalColumnWithIdentity import CategoricalColumnWithIdentity
from torchrec.model import IModel
from torchrec.utils.argument import ArgumentDescription


class FunkSVD(IModel):
    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        argument_descriptions = super().get_argument_descriptions()
        argument_descriptions.extend([
            ArgumentDescription(name="emb_size", type_=int, help_info="Embedding层维度",
                                default_value=64,
                                lower_closed_bound=1),
        ])
        return argument_descriptions

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        super().check_argument_values(arguments)

    def __init__(self,
                 uid_column: CategoricalColumnWithIdentity,
                 iid_column: CategoricalColumnWithIdentity,
                 label_column: CategoricalColumnWithIdentity,
                 emb_size: int,
                 **kwargs):
        self.uid_column = uid_column
        self.iid_column = iid_column
        self.label_column = label_column
        self.emb_size = emb_size
        super().__init__(**kwargs)

    def _init_weights(self):
        self.u_embeddings = Embedding(self.uid_column.category_num, self.emb_size)
        self.i_embeddings = Embedding(self.iid_column.category_num, self.emb_size)

    def forward(self, data: Dict[str, Tensor]):
        u_ids: Tensor = self.uid_column.get_feature_data(data)  # [batch_size]
        i_ids: Tensor = self.iid_column.get_feature_data(data)  # [batch_size] or [batch_size, sample_n]

        u_vectors: Tensor = self.u_embeddings(u_ids)  # [batch_size, emb_size]
        i_vectors: Tensor = self.i_embeddings(i_ids)  # [batch_size, emb_size] or [batch_size, sample_n, emb_size]

        if len(i_ids.shape) == 1:
            prediction: Tensor = (u_vectors * i_vectors).sum(dim=-1)  # [batch_size]

            target = self.label_column.get_feature_data(data)
            if target is not None:
                target = target.float()
        else:
            sample_n: int = i_ids.shape[1]
            # [batch_size * sample_n, emb_size]
            u_vectors: Tensor = u_vectors.unsqueeze(1).repeat(1, sample_n, 1).reshape(-1, self.emb_size)
            # [batch_size * sample_n, emb_size]
            i_vectors: Tensor = i_vectors.reshape(-1, self.emb_size)
            prediction: Tensor = (u_vectors * i_vectors).sum(dim=-1).reshape(-1, sample_n)  # [batch_size, sample_n]

            target = torch.zeros_like(prediction, dtype=torch.float32)
            target[:, 0] = 1

        return prediction, target
