from tensordict import TensorDict
from torch import Tensor


def print_td_shape(td: TensorDict, indent: int = 0):
    indent_str = "\t" * indent
    for k, v in td.items():
        print(f"{indent_str}{k}:", end="")
        if isinstance(v, TensorDict):
            print()
            print_td_shape(v, indent=indent + 1)
        else:
            assert isinstance(v, Tensor)
            print(v.shape)
