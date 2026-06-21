import torch




############################################################
# INITIAL CACHE CREATION
############################################################

import torch

from src.evit.merge import bipartite_soft_matching


def reuse(prototypes, x, mask_reuse, idx):

    print(mask_reuse.shape, idx.shape)

    idx_reuse = idx.where(mask_reuse, torch.zeros_like(idx))  # Set non-reuse indices to 0 (or any valid index, since we'll mask them out later)
    x_reuse = torch.gather(
        x, dim=1, index=idx_reuse.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    )

    k_reuse = torch.gather(
        prototypes, dim=1, index=idx_reuse.unsqueeze(-1).expand(-1, -1, prototypes.shape[-1])
    )   

    v_reuse = torch.gather(
        prototypes, dim=1, index=idx_reuse.unsqueeze(-1).expand(-1, -1, prototypes.shape[-1])
    )

    return x_reuse, k_reuse, v_reuse

def reduce(prototypes, mask_reduce, idx):
    
    idx_reduce = idx.where(mask_reduce, torch.zeros_like(idx))  # Set non-reduce indices to 0 (or any valid index, since we'll mask them out later)
    x_proto = torch.gather(
        prototypes, dim=1, index=idx_reduce.unsqueeze(-1).expand(-1, -1, prototypes.shape[-1])
    )

    k_proto = torch.gather(
        prototypes, dim=1, index=idx_reduce.unsqueeze(-1).expand(-1, -1, prototypes.shape[-1])
    )   

    v_proto = torch.gather(
        prototypes, dim=1, index=idx_reduce.unsqueeze(-1).expand(-1, -1, prototypes.shape[-1])
    )

    return x_proto, k_proto, v_proto
    


def caching(
    x,
    k,
    v,
    cache,
    merge_ratio,
):

    """
    Warmup frame.

    Create prototypes using bipartite soft matching.

    x:
        [B,N,C]

    k,v:
        [B,H,N,D]

    """

    B,H,N,D = k.shape




    # flatten heads

    k_flat = (
        k.transpose(1,2)
        .reshape(
            B,
            N,
            H*D
        )
    )


    v_flat = (
        v.transpose(1,2)
        .reshape(
            B,
            N,
            H*D
        )
    )



    # Use the bipartite soft matching as a clustering algorithm to find prototypes and assign tokens to prototypes


    m,u = bipartite_soft_matching(
        x,
        r=merge_ratio
    )


    prototypes = m(x)


    K_proto_flat = m(k_flat)

    V_proto_flat = m(v_flat)



    M = prototypes.shape[1]



    K_proto = (
        K_proto_flat
        .reshape(
            B,
            M,
            H,
            D
        )
        .transpose(
            1,2
        )
    )


    V_proto = (
        V_proto_flat
        .reshape(
            B,
            M,
            H,
            D
        )
        .transpose(
            1,2
        )
    )



    cache.initialize(
        prototypes,
        K_proto,
        V_proto
    )


    return cache








def matching(
    x,
    k,
    v,
    cache,
):
    

    similarity,idx = cache.match_tokens(x)


    confidence = cache.confidence_score(
        similarity,
        idx
    )


    mask_reuse, mask_reduce, mask_unm = cache.decide(confidence)

    x_reuse, k_reuse, v_reuse = reuse(cache.prototypes, x, mask_reuse, idx)

    unm_idx = idx.where(mask_unm, torch.zeros_like(idx))  # Set non-unmatched indices to 0 (or any valid index, since we'll mask them out later)

    x_unm_idx_expanded = unm_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # [B, num_unm, C]
    x_unm = torch.gather(x, dim=1, index=x_unm_idx_expanded)

    x_proto, k_proto, v_proto = reduce(cache.prototypes, mask_reduce, idx)

    # cache.update(
    #     x,
    #     k,
    #     v,
    #     idx,
    #     similarity
    # )


    return x_unm, x_reuse, x_proto, k_reuse, v_reuse, k_proto, v_proto