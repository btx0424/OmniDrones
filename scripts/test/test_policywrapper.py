from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import VmapModule

import torch


def main():
    lam = TensorDictModule(lambda x: x[0], in_keys=["x"], out_keys=["y"])
    sample_in = torch.ones((10, 3, 2))
    sample_in_td = TensorDict({"x": sample_in}, batch_size=[10])
    print(lam(sample_in).shape)
    vm = VmapModule(lam, 0)
    vm(sample_in_td)
    print(sample_in_td)
    assert (sample_in_td["x"][:, 0] == sample_in_td["y"]).all()

if __name__ == "__main__":
    main()
