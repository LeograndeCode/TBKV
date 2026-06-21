import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeCache(nn.Module):

    """
    Temporal Prototype KV Memory

    Stores semantic prototypes and their corresponding KV states.

    prototypes:
        [B,M,C]

    cache_K:
        [B,H,M,Dh]

    cache_V:
        [B,H,M,Dh]
    """

    def __init__(
        self,
        reuse_threshold=0.85,
        reduce_threshold=0.95,
    ):

        super().__init__()

        self.reuse_threshold = reuse_threshold
        self.reduce_threshold = reduce_threshold

        # memory
        self.prototypes = None
        self.cache_K = None
        self.cache_V = None

        # statistics
        self.hits = None
        self.observations = None
        self.frequency = None
        self.age = None

        # additional defaults
        self.momentum = 0.1
        self.decay = 0.99
        self.num_prototypes = None


    ############################################################
    # INITIALIZATION
    ############################################################


    def initialize(
        self,
        prototypes,
        K,
        V,
    ):

        """
        prototypes: [B,M,C]
        K,V: [B,H,M,D]
        """

        self.prototypes = prototypes.detach()
        self.cache_K = K.detach()
        self.cache_V = V.detach()

        B, M, _ = prototypes.shape
        self.num_prototypes = M
        device = prototypes.device

        self.hits = torch.zeros(B, M, device=device)
        self.observations = torch.zeros(B, M, device=device)
        self.frequency = torch.zeros(B, M, device=device)
        self.age = torch.zeros(B, M, device=device)


    ############################################################
    # MATCHING
    ############################################################


    def match_tokens(self, x):
        """
        x: [B,N,C]
        returns: similarity [B,N], idx [B,N]
        """
        a = x / x.norm(dim=-1, keepdim=True)  # [B, N, C]
        b = self.prototypes / self.prototypes.norm(dim=-1, keepdim=True)  # [B, M, C]

        similarity = a @ b.transpose(-1, -2)  # [B, N, M]

        best_similarity, idx = similarity.max(dim=-1)
        return best_similarity, idx


    ############################################################
    # CONFIDENCE
    ############################################################


    def confidence_score(self, similarity, idx):
        stability = self.hits / (self.observations + 1e-6)
        proto_stability = torch.gather(stability, 1, idx)
        confidence = similarity * proto_stability
        return confidence


    ############################################################
    # DECISION
    ############################################################


    def decide(self, confidence):
        reuse = confidence > self.reuse_threshold
        reduce = confidence > self.reduce_threshold
        unmatched = confidence <= self.reuse_threshold
        return reuse, reduce, unmatched






    ############################################################
    # RETRIEVE KV
    ############################################################


    def get_cached_kv(
        self,
        idx
    ):

        """

        idx:
            [B,N]


        returns:

        K:
            [B,H,N,D]

        V:
            [B,H,N,D]

        """


        B,N = idx.shape


        H = self.cache_K.shape[1]

        D = self.cache_K.shape[-1]



        index = idx[:,None,:,None]


        index = index.expand(
            B,
            H,
            N,
            D
        )



        K = torch.gather(
            self.cache_K,
            2,
            index
        )


        V = torch.gather(
            self.cache_V,
            2,
            index
        )


        return K,V





    ############################################################
    # UPDATE MEMORY
    ############################################################


    @torch.no_grad()
    def update(
        self,
        x,
        K,
        V,
        idx,
        similarity
    ):

        """

        x:
            [B,N,C]

        K,V:
            [B,H,N,D]

        idx:
            [B,N]


        """

        B,N,C = x.shape


        H = K.shape[1]

        D = K.shape[-1]



        # temporal aging

        self.age += 1



        for b in range(B):

            for m in range(self.num_prototypes):


                mask = (
                    idx[b]==m
                )


                if mask.sum()==0:
                    continue



                tokens = x[b,mask]

                new_proto = tokens.mean(
                    dim=0
                )



                # prototype EMA

                self.prototypes[b,m] = (
                    (1-self.momentum)
                    *
                    self.prototypes[b,m]
                    +
                    self.momentum
                    *
                    new_proto
                )



                ################################################
                # KV update
                ################################################


                new_K = K[b,:,mask,:].mean(
                    dim=1
                )

                new_V = V[b,:,mask,:].mean(
                    dim=1
                )



                self.cache_K[b,:,m,:] = (
                    (1-self.momentum)
                    *
                    self.cache_K[b,:,m,:]
                    +
                    self.momentum
                    *
                    new_K
                )


                self.cache_V[b,:,m,:] = (
                    (1-self.momentum)
                    *
                    self.cache_V[b,:,m,:]
                    +
                    self.momentum
                    *
                    new_V
                )




                ################################################
                # statistics
                ################################################


                self.observations[b,m]+=1



                if similarity[b,mask].mean() > self.reuse_threshold:

                    self.hits[b,m]+=1



                self.frequency[b,m] = (
                    self.decay*
                    self.frequency[b,m]
                    +
                    (1-self.decay)
                )


                self.age[b,m]=0





    ############################################################
    # REMOVE UNUSED PROTOTYPES INFO
    ############################################################


    def dead_prototypes(
        self,
        threshold=1e-4
    ):

        return (
            self.frequency <
            threshold
        )



    @property
    def stability(self):

        return (
            self.hits /
            (self.observations+1e-6)
        )

        



def main():

    torch.manual_seed(0)

    # -----------------------------
    # Fake ViT parameters
    # -----------------------------

    B = 2          # batch size
    N = 196        # tokens per image
    C = 64         # embedding dimension
    M = 16         # number of prototypes
    H = 4          # number of heads
    D = 32         # KV head dimension


    # -----------------------------
    # Create cache
    # -----------------------------

    cache = PrototypeCache(reuse_threshold=0.75, reduce_threshold=0.90)


    # -----------------------------
    # Warmup phase
    # create initial prototypes
    # -----------------------------

    prototypes = torch.randn(
        B,
        M,
        C
    )

    K_cache = torch.randn(
        B,
        H,
        M,
        D
    )

    V_cache = torch.randn(
        B,
        H,
        M,
        D
    )


    cache.initialize(
        prototypes,
        K_cache,
        V_cache
    )


    print("\n=== INITIAL CACHE ===")
    print("Prototypes:",
          cache.prototypes.shape)

    print("K cache:",
          cache.cache_K.shape)



    # -----------------------------
    # New incoming frame
    # -----------------------------

    x = torch.randn(
        B,
        N,
        C
    )


    print("\n=== INPUT FRAME ===")
    print("Tokens:",
          x.shape)



    # -----------------------------
    # Matching
    # -----------------------------

    similarity, idx = cache.match_tokens(x)


    print("\n=== MATCHING ===")

    print(
        "Similarity shape:",
        similarity.shape
    )

    print(
        "Prototype index shape:",
        idx.shape
    )


    print(
        "Average similarity:",
        similarity.mean().item()
    )

    print(
        "Best similarity:",
        similarity.max().item()
    )



    # -----------------------------
    # Confidence
    # -----------------------------

    confidence = cache.confidence_score(
        similarity,
        idx
    )


    print("\n=== CONFIDENCE ===")

    print(
        "Average confidence:",
        confidence.mean().item()
    )


    # -----------------------------
    # Decisions
    # -----------------------------

    reuse, reduce, _ = cache.decide(confidence)


    print("\n=== DECISION ===")

    print(
        "Reusable tokens:",
        reuse.sum().item(),
        "/",
        B*N
    )


    print(
        "Reducible tokens:",
        reduce.sum().item(),
        "/",
        B*N
    )


    # -----------------------------
    # Get cached KV
    # -----------------------------

    K,V = cache.get_cached_kv(idx)


    print("\n=== KV RETRIEVAL ===")

    print(
        "Retrieved K:",
        K.shape
    )

    print(
        "Retrieved V:",
        V.shape
    )



    print("\n=== STATISTICS ===")
    try:
        print("Mean hit rate:", float(cache.stability.mean()))
        print("Mean frequency:", float(cache.frequency.mean()))
    except Exception:
        print("Statistics not available")





if __name__ == "__main__":
    main()

