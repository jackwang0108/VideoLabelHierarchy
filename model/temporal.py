# Standard Library
from typing import Optional

# Torch Library
import torch
import torch.nn as nn


class FCPrediction(nn.Module):

    def __init__(
        self,
        # input feature dimension
        feature_dim: int,
        # output classes dimension
        num_classes: int,
    ) -> None:
        """
        fully connected classifier

        when calling:
            input shape: [Batch, Temporal, Feature]
            output shape: [Batch, Temporal, Classes]

        Args:
            feature_dim (int): input feature dimension.
            num_classes (int): output classes dimension.
        """
        super().__init__()
        self.fc = nn.Linear(in_features=feature_dim, out_features=num_classes)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # sourcery skip: inline-immediately-returned-variable

        # x: [Batch, Temporal, Feature]
        batch_size, clip_len, feature_dim = x.size()
        # logits: [Batch, Temporal, Classes]
        logits = self.fc(x.reshape(batch_size * clip_len, -1)
                         ).view(batch_size, clip_len, -1)
        return logits


class VanillaGRUPrediction(nn.Module):

    def __init__(
        self,
        # input feature dimension
        feature_dim: int,
        # output classes dimension
        num_classes: int,
        # hidden feature dimension inside GRU
        hidden_dim: int,
        # number of layers in GRU
        num_layers: Optional[int] = 1
    ) -> None:
        """
        vanilla GRU classifier 

        when calling:
            input shape: [Batch, Temporal, Feature]
            output shape: [Batch, Temporal, Classes]

        Args:
            feature_dim (int): size of features 
            num_classes (int): _description_
            hidden_dim (int): _description_
            num_layers (Optional[int], optional): _description_. Defaults to 1.
        """
        super().__init__()

        self.gru = nn.GRU(
            feature_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout()

        # single level prediction head
        self.fc = FCPrediction(2 * hidden_dim, num_classes)

    def forward(self, feature: torch.FloatTensor) -> torch.FloatTensor:
        # sourcery skip: inline-immediately-returned-variable

        # x: [Batch, Temporal, Feature]
        # temporalized_feature: [Batch, Temporal, 2 * Feature], 2 * Feature for bidirectional GRU
        temporalized_feature, _ = self.gru(feature)

        # logits: [Batch, Temporal, Classes]
        logits = self.fc(self.dropout(temporalized_feature))
        return {0: logits}


class UniTransGRUPrediction(nn.Module):

    def __init__(
        self,
        # input feature dimension
        feature_dim: int,
        # output classes dimension
        num_classes: list[int],
        # hidden feature dimension inside GRU
        hidden_dim: int,
        # number of layers in GRU
        num_layers: Optional[int] = 1,
        start_epoch: int = -1
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            feature_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout()

        # prediction heads
        self.fc: list[FCPrediction] = [
            FCPrediction(2 * hidden_dim, nc) for nc in num_classes]

        # Transition MLP
        self.row_mlp: list[nn.Linear] = []
        self.col_mlp: list[nn.Linear] = []

        self.levels = len(num_classes) - 1
        for col_size, row_size in zip(num_classes[:-1], num_classes[1:]):
            self.row_mlp.append(nn.Linear(2 * hidden_dim, row_size))
            self.col_mlp.append(nn.Linear(2 * hidden_dim, col_size))

        self.start_epoch: int = start_epoch

    def get_transition_logits(
        self, feature: torch.FloatTensor,
        classification_logits: list[torch.FloatTensor],
        level: int
    ) -> torch.FloatTensor:
        # sourcery skip: inline-immediately-returned-variable

        # [Batch, Temporal, Row]
        trans_row = self.row_mlp[level](feature)
        # [Batch, Temporal, Col]
        trans_col = self.col_mlp[level](feature)
        trans_matrix = torch.einsum("btr,btc->btrc", trans_row, trans_col)
        trans_logits = torch.einsum(
            "btrc,btc->btr", trans_matrix, classification_logits[level])
        return trans_logits

    def forward(self, feature: torch.FloatTensor, epoch: int) -> dict[str, torch.FloatTensor]:
        # x: [Batch, Temporal, Feature]
        # temporalized_feature: [Batch, Temporal, 2 * Feature], 2 * Feature for bidirectional GRU
        temporalized_feature, _ = self.gru(feature)

        classification_logits = [fc(temporalized_feature) for fc in self.fc]

        transition_logits = []
        for i in range(self.levels):
            if epoch >= self.start_epoch:
                transition_logits.append(self.get_transition_logits(
                    temporalized_feature, classification_logits, i))
            else:
                transition_logits.append(
                    torch.zeros_like(classification_logits[i+1]))

        results = {0: classification_logits[0]}
        for i in range(self.levels):
            results[i+1] = classification_logits[i+1] + transition_logits[i]

        return results


class BiTransGRUPrediction(nn.Module):
    # TODO: finish BiTransGRUPrediction
    pass


if __name__ == "__main__":
    from pathlib import Path
    from datasets.image_datasets import get_hierarchal_classes

    classes = get_hierarchal_classes(
        Path(__file__).parent / "../tools/tennis/class.yaml")

    import pprint

    num_classes = sorted(
        [len(v) for v in classes["level_dict"].values()], reverse=False)

    ugp = UniTransGRUPrediction(512, num_classes, 736, start_epoch=10)

    feature = torch.randn(4, 100, 512)
    ugp(feature, 0)
