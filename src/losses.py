import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class NegativeSoftPlusLoss(_Loss):
    """
    Negative softplus loss used in TriVec model.
    """
    def __init__(self) -> None:
        super(NegativeSoftPlusLoss, self).__init__()

    def forward(self, pos_scores: torch.Tensor,
                neg_scores: torch.Tensor) -> torch.Tensor:
        input_scores = torch.cat((pos_scores * -1, neg_scores))
        return torch.mean(F.softplus(input_scores))
