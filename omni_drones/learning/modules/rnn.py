import torch
import torch.nn as nn

"""
These modules are walk-arounds for using functorch.vmap since the batching rules for
GRU and LSTM are not implemented yet (2023/03/13).
"""

class GRU(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int
    ) -> None:
        super().__init__()
        self.cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        nn.init.orthogonal_(self.cell.weight_hh)
        nn.init.orthogonal_(self.cell.weight_ih)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        input: torch.Tensor,
        h: torch.Tensor = None,
        is_initial: torch.Tensor = None,
    ):
        """
        input: [N, L, H_in] or [N, H_in]
        """
        
        if input.dim() == 3:
            has_time_dim = True
            N, L = input.shape[:2]
        else:
            has_time_dim = False
            N = input.shape[0]

        if h is None:
            h = torch.zeros(N, self.cell.hidden_size, device=input.device)
        elif h.dim() > 2 and has_time_dim:
            h = h[:, 0]
        
        
        if has_time_dim:
            if is_initial is None:
                is_initial = torch.zeros(N, L, 1, device=input.device)
            mask = (1 - is_initial.float()).reshape(N, L, 1)
            output = []
            for i in range(L):
                h = h * mask[:, i]
                h = self.cell(input[:, i], h)
                output.append(h.clone())
            output = torch.stack(output, dim=1)
        else:
            if is_initial is None:
                is_initial = torch.zeros(N, 1, device=input.device)
            mask = (1 - is_initial.float()).reshape(N, 1)
            output = h = self.cell(input, h * mask)

        # output = output + input # 0
        output = self.layer_norm(output + input) # 1
        # output = self.layer_norm(output) + input # 2
        # output = self.layer_norm(output) # 3
        if has_time_dim:
            h = h.unsqueeze(1).expand(N, L, -1)  # pad to the same length
        return output, h


# class LSTM(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int):
#         super().__init__()
#         self.cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

#     def forward(
#         self,
#         input: torch.Tensor,
#         hc: torch.Tensor = None,
#         is_initial: torch.Tensor = None,
#     ):
#         assert input.dim() in (2, 3)
#         L = input.shape[0]
#         if input.dim() == 3:
#             L, N = input.shape[:2]
#         else:
#             L, N = input.shape[0], 1
#             input = input.unsqueeze(1)

#         if hc is None:
#             h = torch.zeros(N, self.cell.hidden_size, device=input.device)
#             c = torch.zeros(N, self.cell.hidden_size, device=input.device)
#         else:
#             h, c = hc.chunk(2, dim=-1)

#         if is_initial is None:
#             is_initial = torch.zeros(L, N, device=input.device)
#         mask = 1 - is_initial.float()

#         outputs = []
#         for i in range(L):
#             h = h * mask[i]
#             c = c * mask[i]
#             (h, c) = self.cell(input[i], (h, c))
#             outputs.append(h.clone())
#         outputs = torch.stack(outputs)
#         hc = torch.cat([h, c], dim=-1).expand(L, -1)
#         return outputs, hc  # pad to the same length
