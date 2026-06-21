import torch
from typing import Tuple
from einops import rearrange
from typing import Any, Callable, Dict, Optional, Tuple
from src.evit.merge import bipartite_soft_matching



def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False

def init_generator(device: torch.device, fallback: torch.Generator=None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback

def join_frame(x, fsize):
    """ Join multi-frame tokens """
    x = rearrange(x, "(B F) N C -> B (F N) C", F=fsize)
    return x

def split_frame(x, fsize):
    """ Split multi-frame tokens """
    x = rearrange(x, "B (F N) C -> (B F) N C", F=fsize)
    return x

def func_warper(funcs):
    """ Warp a function sequence """
    def fn(x, **kwarg):
        for func in funcs:
            x = func(x, **kwarg)
        return x
    return fn

def join_warper(fsize):
    def fn(x):
        x = join_frame(x, fsize)
        return x
    return fn

def split_warper(fsize):
    def fn(x):
        x = split_frame(x, fsize)
        return x
    return fn


def get_uniques_indexes_padded(idx: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    """
    Get unique indexes for each batch and pad to the minimum number of unique indexes across the batch using confidence as sorting key.
    """
    B, N = idx.shape

    unique_idx_list = []
    max_unique = 0
    for b in range(B):
        unique_idx = torch.unique(idx[b])
        unique_idx_list.append(unique_idx)
        max_unique = max(max_unique, len(unique_idx))

    # Pad unique indexes to the maximum number of unique indexes across the batch
    padded_unique_idx = torch.zeros((B, max_unique), dtype=idx.dtype, device=idx.device)
    for b in range(B):
        unique_idx = unique_idx_list[b]
        conf_b = confidence[b]
        # Sort unique indexes by confidence
        sorted_unique_idx = unique_idx[torch.argsort(conf_b[unique_idx], descending=True)]
        padded_unique_idx[b, :len(sorted_unique_idx)] = sorted_unique_idx

    return padded_unique_idx
