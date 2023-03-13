import torch
import torch.nn as nn

"""
These modules are walk-arounds for using functorch.vmap since the batching rules for
GRU and LSTM are not implemented yet (2023/03/13).
"""

class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

    def forward(
        self,
        input: torch.Tensor,
        h: torch.Tensor = None,
        is_initial: torch.Tensor = None,
    ):
        assert input.dim() in (2, 3)
        if input.dim() == 3:
            L, N = input.shape[:2]
        else:
            L, N = input.shape[0], 1
            input = input.unsqueeze(1)

        if h is None:
            h = torch.zeros(N, self.cell.hidden_size, device=input.device)
        if is_initial is None:
            is_initial = torch.zeros(L, N, 1, device=input.device)
        mask = (1 - is_initial.float()).reshape(L, N, 1)

        outputs = []
        for i in range(L):
            h = h * mask[i]
            h = self.cell(input[i], h)
            outputs.append(h.clone())
        outputs = torch.stack(outputs)
        return outputs, h.expand(L, N, -1)  # pad to the same length


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def forward(
        self,
        input: torch.Tensor,
        hc: torch.Tensor = None,
        is_initial: torch.Tensor = None,
    ):
        assert input.dim() in (2, 3)
        L = input.shape[0]
        if input.dim() == 3:
            L, N = input.shape[:2]
        else:
            L, N = input.shape[0], 1
            input = input.unsqueeze(1)

        if hc is None:
            h = torch.zeros(N, self.cell.hidden_size, device=input.device)
            c = torch.zeros(N, self.cell.hidden_size, device=input.device)
        else:
            h, c = hc.chunk(2, dim=-1)

        if is_initial is None:
            is_initial = torch.zeros(L, N, device=input.device)
        mask = 1 - is_initial.float()

        outputs = []
        for i in range(L):
            h = h * mask[i]
            c = c * mask[i]
            (h, c) = self.cell(input[i], (h, c))
            outputs.append(h.clone())
        outputs = torch.stack(outputs)
        hc = torch.cat([h, c], dim=-1).expand(L, -1)
        return outputs, hc  # pad to the same length
