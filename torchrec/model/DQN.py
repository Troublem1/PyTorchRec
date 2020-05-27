from typing import Dict, Any, List

from torchrec.model.IValueRLModel import IValueRLModel
from torchrec.utils.argument import ArgumentDescription


class DQN(IValueRLModel):
    def forward(self, *input):
        pass

    @classmethod
    def get_argument_descriptions(cls) -> List[ArgumentDescription]:
        pass

    @classmethod
    def check_argument_values(cls, arguments: Dict[str, Any]) -> None:
        pass
