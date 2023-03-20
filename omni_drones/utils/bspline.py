import torch
import numpy as np
from typing import Optional

from scipy.interpolate import (
    splev as _splev_scipy_impl, 
    splint as _splint_scipy_impl,
)

def splev_scipy(x, t, c, k, der=0):
    """
    Evaluate a B-spline or its derivatives.

    Given the knots and coefficients of a B-spline representation, evaluate
    the value of the smoothing polynomial and its derivatives. This is a
    wrapper around the FORTRAN routines splev and splder of FITPACK.

    Parameters
    ----------
    x : array_like
        An array of points at which to return the value of the smoothed
        spline or its derivatives. If `tck` was returned from `splprep`,
        then the parameter values, u should be given.
    tck : 3-tuple or a BSpline object
        If a tuple, then it should be a sequence of length 3 returned by
        `splrep` or `splprep` containing the knots, coefficients, and degree
        of the spline. (Also see Notes.)
    der : int, optional
        The order of derivative of the spline to compute (must be less than
        or equal to k, the degree of the spline).
    """
    return np.stack(_splev_scipy_impl(x, (t, c.T, k), der), axis=-1)

def splint_scipy(a, b, t, c, k):
    """
    Evaluate the definite integral of a B-spline between two given points.

    Parameters
    ----------
    a, b : float
        The end-points of the integration interval.
    tck : tuple or a BSpline instance
        If a tuple, then it should be a sequence of length 3, containing the
        vector of knots, the B-spline coefficients, and the degree of the
        spline (see `splev`).
    full_output : int, optional
        Non-zero to return optional output.
    """
    return _splint_scipy_impl(a, b, (t, c, k))

def splev_torch(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int, der: int=0):
    """
    Evaluate a B-spline or its derivatives.

    Parameters
    ----------
    x : Tensor
        An array of points at which to return the value of the smoothed
        spline or its derivatives. If `tck` was returned from `splprep`,
        then the parameter values, u should be given.
    t, c, k : Tensor
        If a tuple, then it should be a sequence of length 3 returned by
        `splrep` or `splprep` containing the knots, coefficients, and degree
        of the spline. (Also see Notes.)
    der : int, optional
        The order of derivative of the spline to compute (must be less than
        or equal to k, the degree of the spline).
    """
    if der == 0:
        # return _splev_torch_batched_impl(x, t, c, k)
        return _splev_torch_impl(x, t, c, k)
    else:
        assert der <= k, "The order of derivative to compute must be less than or equal to k."
        n = c.size(-2)
        return k * splev_torch(x, t[..., 1:-1], (c[...,1:,:]-c[...,:-1,:])/(t[k+1:k+n]-t[1:n]).unsqueeze(-1), k-1, der-1)

def _splev_torch_impl(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int):
    """
        x: (nx,)
        t: (m, )
        c: (n, dim)
    """
    assert t.size(0) == c.size(0) + k + 1, f"{len(t)} != {len(c)} + {k} + {1}" # m= n + k + 1

    x = torch.atleast_1d(x)
    assert x.dim() == 1 and t.dim() == 1 and c.dim() == 2, f"{x.shape}, {t.shape}, {c.shape}"
    n = c.size(0)
    u = (torch.searchsorted(t, x)-1).clip(k, n-1).unsqueeze(-1)
    x = x.unsqueeze(-1)
    d = c[u-k+torch.arange(k+1, device=c.device)].contiguous()
    for r in range(1, k+1):
        j = torch.arange(r-1, k, device=c.device) + 1
        t0 = t[j+u-k]
        t1 = t[j+u+1-r]
        alpha = ((x - t0) / (t1 - t0)).unsqueeze(-1)
        d[:, j] = (1-alpha)*d[:, j-1] + alpha*d[:, j]
    return d[:, k]
    
def init_traj(
    start_pos: torch.Tensor, 
    end_pos: torch.Tensor, 
    start_vel: Optional[torch.Tensor]=None, 
    start_acc: Optional[torch.Tensor]=None,
    end_vel: Optional[torch.Tensor]=None,
    env_acc: Optional[torch.Tensor]=None,
    n_ctps: int=10,
    k: int=3,
):
    """

    pos: (*batch, dim)
    linvel: (*batch, dim)
    target_pos: (*batch, dim)
    
    n_ctps: number of control points
    k: degree of b-spline
    """
    assert n_ctps >= 6

    device = start_pos.device
    if start_vel is None:
        start_vel = torch.zeros_like(start_pos)
    if start_acc is None:
        start_acc = torch.zeros_like(start_pos)
    if end_vel is None:
        end_vel = torch.zeros_like(start_vel)
    if env_acc is None:
        env_acc = torch.zeros_like(start_acc)
    
    assert start_pos.shape == end_pos.shape
    
    ctps_0 = start_pos
    ctps_1 = start_pos + start_vel / k
    ctps_2 = start_pos + start_vel + start_acc / k

    ctps_n = end_pos
    ctps_n_1 = end_pos - end_vel / k
    ctps_n_2 = end_pos - end_vel - env_acc / k

    ctps_inter = (
        ctps_2.unsqueeze(-2) 
        + (ctps_n_2 - ctps_2).unsqueeze(-2) 
        * torch.linspace(0, 1, n_ctps-4, device=device).unsqueeze(-1)
    )
    ctps = torch.cat([
        ctps_0.unsqueeze(-2),
        ctps_1.unsqueeze(-2),
        ctps_inter,
        ctps_n_1.unsqueeze(-2),
        ctps_n.unsqueeze(-2),
    ], dim=-2)
    knots = torch.cat([
        torch.zeros(k, device=device), 
        torch.arange(n_ctps+1-k, device=device), 
        torch.full((k,), n_ctps-k, device=device),
    ])
    return ctps, knots

def get_ctps(
    c: torch.Tensor, 
    x: torch.Tensor, 
    start:int=3,
    end:int=-3,
):
    """
    reshape the decision var x and plug it into the control points at start:end

    c: control points of shape [nctps, dim]
    x: decision var of shape [(n-start-end), dim] or [(n-start-end) * dim, ]
    """
    assert c.dim() == 2
    nctps, dim = c.shape

    end = end % nctps
    x = x.reshape(end-start, dim)
    c_start = c[:start]
    c_end = c[end:]
    c = torch.cat([c_start, x, c_end])
    return c

def get_knots(n_ctps: int, k: int, device="cpu"):
    knots = torch.cat([
        torch.zeros(k, device=device), 
        torch.arange(n_ctps+1-k, device=device), 
        torch.full((k,), n_ctps-k, device=device),
    ])
    return knots