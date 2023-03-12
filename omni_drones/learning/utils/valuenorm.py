from typing import Iterable, Union

import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def update(self, input_vector: torch.Tensor):
        ...

    def normalize(self, input_vector: torch.Tensor):
        ...

    def denormalize(self, input_vector: torch.Tensor):
        ...


class ValueNorm1(Normalizer):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out


class ValueNorm2(Normalizer):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        subtract_mean: bool = True,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.subtract_mean = subtract_mean
        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor
        self.count: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_var", torch.ones(input_shape))
        self.register_buffer("count", torch.tensor(0))
        self.eps = torch.finfo(torch.float32).eps

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_var = input_vector.var(dim=dim)
        batch_count = input_vector.shape[: -len(self.input_shape)].numel()

        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count

        self.running_mean.add_(delta * batch_count / total_count)
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count
        self.running_var[:] = new_var

        self.count.add_(batch_count)

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        if self.subtract_mean:
            return (input_vector - self.running_mean) / torch.sqrt(
                self.running_var + self.eps
            )
        else:
            return input_vector / torch.sqrt(self.running_var + self.eps)

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        if self.subtract_mean:
            return input_vector * torch.sqrt(self.running_var) + self.running_mean
        else:
            return input_vector * torch.sqrt(self.running_var)
