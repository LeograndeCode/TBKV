import torch
from typing import Tuple
from einops import rearrange
from typing import Any, Callable, Dict, Optional, Tuple
from src.tbkv.merge import bipartite_soft_matching

def get_objective_score(score_attn):
    """Compute saliency score from attention."""
    # Average over attention heads if present: [B, num_heads, N, N] -> [B, N, N]
    if score_attn.dim() == 4:
        score_attn = score_attn.mean(dim=1)
    
    # Compute entropy over the attention distribution (last dimension)
    scores = (score_attn * torch.log(score_attn + 1e-6)).sum(dim=-1).unsqueeze(-1)

    # foreground removal (normalize across tokens, not time)
    scores = scores - scores.amin(dim=-2, keepdim=True)
    scores = scores / (scores.amax(dim=-2, keepdim=True) + 1e-6)
    score_mask = scores >= scores.mean(dim=-2, keepdim=True)

    # background sharpening
    scores = scores - scores.mean(dim=-2, keepdim=True)
    scores = scores / (scores.amax(dim=-2, keepdim=True) + 1e-6)
    scores[score_mask] = 0.0
    return scores


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


def extract_bg_fg_tokens(x: torch.Tensor, attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Extract background and foreground tokens based on attention map."""

    B, N, C = x.shape


    # Get saliency score for each token so we can separate fg/bg

    scores = get_objective_score(attn).squeeze(-1) # [B, N]

    # compute max bg/fg tokens count

    max_bg_tokens = (scores == 0).sum(dim=-1).max().item()
    max_fg_tokens = (scores != 0).sum(dim=-1).max().item()

    # Get bg and foreground indicies following topk approach
    # Basically we allow some "usure foreground" tokes to be 
    # considered as background to have fixed size tensors and viceversa
    # This mechanism additionally allows to handle an error margins in the saliency map
    # by allowing some fg tokens to be considered as bg and viceversa
    
    idx_bg = torch.topk(scores, dim=-1, k=max_bg_tokens, largest=False).indices  # [B, max_bg_tokens]
    idx_fg = torch.topk(scores, dim=-1, k=max_fg_tokens, largest=True).indices   # [B, max_fg_tokens]

    x_bg = torch.gather(x, dim=-2, index=idx_bg.unsqueeze(-1).expand(-1, -1, C))  # [B, max_bg_tokens, C]
    x_fg = torch.gather(x, dim=-2, index=idx_fg.unsqueeze(-1).expand(-1, -1, C))  # [B, max_fg_tokens, C]

    return x_bg, x_fg, idx_bg, idx_fg


    
def compute_merge( x: torch.Tensor, local_merge_ratio: float) -> Tuple[Callable, ...]:
    """
    Token merging for VideoMAE.
    Same as timm_TBKV but adapted for VideoMAE's architecture.
    
    :param module: VideoMAE Block module
    :param x: Tokens to be merged [B, N, C]
    :param tome_info: Dictionary for merging configuration
    :return: Returns merge function, unmerge function, and merged tokens.
    """


    # Apply bipartite soft matching

    m, u = bipartite_soft_matching(x, local_merge_ratio, class_token=False, distill_token=False)

    original_tokens = x
    merged_tokens = m(x)

    #Debug
    print(f"Merge info - merge_ratio: {local_merge_ratio}, original tokens: {original_tokens.shape[1]}, merged tokens: {merged_tokens.shape[1]}")
        
    # # Global Token Merging!
    # if args["merge_global"]:
    #     if hasattr(module, "global_tokens") and module.global_tokens is not None:
    #         if torch.rand(1, generator=generator, device=generator.device) > args["global_rand"]:
    #             src_len = local_tokens.shape[1]
    #             tokens = torch.cat([local_tokens, module.global_tokens.to(local_tokens)], dim=1)
    #             local_chunk = 0
    #         else:
    #             src_len = module.global_tokens.shape[1]
    #             tokens = torch.cat([module.global_tokens.to(local_tokens), local_tokens], dim=1)
    #             local_chunk = 1

    #         m, u, _ = merge.bipartite_soft_matching_2s(
    #             tokens, src_len, args["global_merge_ratio"], merge_mode=merge_mode, unmerge_chunk=local_chunk)
    #         merged_tokens = m(tokens)
    #         m_ls.append(m)
    #         u_ls.append(u)
    #         module.global_tokens = u(merged_tokens).detach().clone().cpu()
    #     else:
    #         module.global_tokens = local_tokens.detach().clone().cpu()

    #     m = func_warper(m_ls)
    #     u = func_warper(u_ls[::-1])
    # else:
    #     m, u = (merge.do_nothing, merge.do_nothing)
    #     merged_tokens = x

    return m, u, merged_tokens