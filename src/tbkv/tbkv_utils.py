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

    # Normalize across tokens (not time) so larger values mean higher saliency.
    scores = scores - scores.amin(dim=-2, keepdim=True)
    scores = scores / (scores.amax(dim=-2, keepdim=True) + 1e-6)

    # Recenter so background tends lower and foreground tends higher, while
    # preserving rank information needed by top-k token selection.
    scores = scores - scores.mean(dim=-2, keepdim=True)
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


def extract_bg_fg_tokens(x: torch.Tensor, attn: torch.Tensor,
                         bg_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Extract background and foreground tokens based on attention map.

    The split is an exact partition: n_bg + n_fg == N. Sizing the two sets
    independently (e.g. from a per-item threshold, then taking the batch max
    of each) makes n_bg + n_fg exceed N whenever batch items disagree on the
    split, which duplicates tokens into both sets and inflates the sequence
    length at every block.
    """

    B, N, C = x.shape

    # Accept either a full attention map [B, heads, N, N] or an already-reduced
    # saliency vector [B, N]. The ViTDet path passes the latter: it collapses
    # attention to compact saliency before propagating it between blocks (a full
    # map is ~150 MB there), so re-scoring it would collapse the token axis.
    if attn.dim() == 2:
        scores = attn.float()
    else:
        scores = get_objective_score(attn).squeeze(-1)  # [B, N]

    # Fixed, batch-uniform split sizes so the two sets tile the sequence.
    n_bg = int(round(N * bg_ratio))
    n_bg = max(0, min(N, n_bg))
    n_fg = N - n_bg

    # Background = lowest saliency, foreground = highest saliency. Ranking by
    # top-k (rather than thresholding at the mean) keeps the sets disjoint and
    # lets borderline tokens fall to whichever side has room.
    idx_bg = torch.topk(scores, dim=-1, k=n_bg, largest=False).indices  # [B, n_bg]
    idx_fg = torch.topk(scores, dim=-1, k=n_fg, largest=True).indices   # [B, n_fg]

    x_bg = torch.gather(x, dim=-2, index=idx_bg.unsqueeze(-1).expand(-1, -1, C))  # [B, n_bg, C]
    x_fg = torch.gather(x, dim=-2, index=idx_fg.unsqueeze(-1).expand(-1, -1, C))  # [B, n_fg, C]

    return x_bg, x_fg, idx_bg, idx_fg


    
def compute_merge(x: torch.Tensor, merging_iterations: int,
                  local_merge_ratio: float = 0.5) -> Tuple[Callable, ...]:

    """
    Token merging for VideoMAE.
    Same as timm_TBKV but adapted for VideoMAE's architecture.

    :param x: Tokens to be merged [B, N, C]
    :param merging_iterations: Number of successive bipartite matching passes.
    :param local_merge_ratio: Fraction of tokens removed per iteration. The
        token count decays as (1 - local_merge_ratio) ** merging_iterations,
        so this must stay small when merging_iterations is large.
    :return: Returns merge function, unmerge function, and merged tokens.
    """

    # Apply bipartite soft matching
    merged_tokens = x
    m_seq = []
    u_seq = []
    for _ in range(merging_iterations):
        m_i, u_i = bipartite_soft_matching(
            merged_tokens,
            local_merge_ratio,
            class_token=False,
            distill_token=False,
        )
        m_seq.append(m_i)
        u_seq.append(u_i)
        merged_tokens = m_i(merged_tokens)

    def m(tokens: torch.Tensor) -> torch.Tensor:
        out = tokens
        for m_i in m_seq:
            out = m_i(out)
        return out

    def u(tokens: torch.Tensor) -> torch.Tensor:
        out = tokens
        for u_i in reversed(u_seq):
            out = u_i(out)
        return out

    original_tokens = x

    #Debug

        
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


def perform_kv_reuse_no_reduction(
    x: torch.Tensor,
    cache_tokens: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    q_proj,
    k_proj,
    v_proj,
    num_heads: int,
    r_match: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Build full-grid Q/K/V with KV reuse only (no token reduction / no fg-bg split).

    This keeps the query/key/value token count fixed to N and only replaces
    matched token K/V with cached values.
    """
    b, n, c = x.shape
    head_dim = c // num_heads

    q = q_proj(x).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
    k = torch.zeros((b, num_heads, n, head_dim), device=x.device, dtype=x.dtype)
    v = torch.zeros((b, num_heads, n, head_dim), device=x.device, dtype=x.dtype)

    # Match each token to cache by cosine similarity, then select top-r matches.
    x_n = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
    cache_n = cache_tokens / (cache_tokens.norm(dim=-1, keepdim=True) + 1e-6)
    sim = torch.matmul(x_n, cache_n.transpose(-2, -1))
    best_sim, best_cache_idx = sim.max(dim=-1)

    n_match = max(0, min(n, int(n * float(r_match))))
    sorted_idx = best_sim.argsort(dim=-1, descending=True)
    idx_match = sorted_idx[:, :n_match]
    idx_unmatch = sorted_idx[:, n_match:]

    if n_match > 0:
        matched_cache_idx = torch.gather(best_cache_idx, 1, idx_match)
        gather_cache = matched_cache_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)
        k_match = torch.gather(cache_k, 2, gather_cache)
        v_match = torch.gather(cache_v, 2, gather_cache)
        scatter_match = idx_match.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)
        k = k.scatter(2, scatter_match, k_match)
        v = v.scatter(2, scatter_match, v_match)

    n_unmatch = idx_unmatch.shape[1]
    if n_unmatch > 0:
        gather_unmatch = idx_unmatch.unsqueeze(-1).expand(-1, -1, c)
        x_unmatch = x.gather(1, gather_unmatch)
        k_new = k_proj(x_unmatch).reshape(b, n_unmatch, num_heads, head_dim).permute(0, 2, 1, 3)
        v_new = v_proj(x_unmatch).reshape(b, n_unmatch, num_heads, head_dim).permute(0, 2, 1, 3)
        scatter_unmatch = idx_unmatch.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)
        k = k.scatter(2, scatter_unmatch, k_new)
        v = v.scatter(2, scatter_unmatch, v_new)

    info = {
        "n_tokens": int(n),
        "n_matched_tokens": int(n_match),
        "n_unmatched_tokens": int(n_unmatch),
    }
    return q, k, v, info
