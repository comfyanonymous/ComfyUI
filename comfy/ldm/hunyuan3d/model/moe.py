import torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:

        if gate.device.type == "mps":
            return F.gelu(gate.to(dtype = torch.float32)).to(dtype = gate.dtype)
        
        return F.gelu(gate)

    def forward(self, hidden_states):

        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)

        return hidden_states

class FeedForward(nn.Module):

    def __init__(self, dim: int, dim_out = None, mult: int = 4,
                dropout: float = 0.0, inner_dim = None):
        
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)

        dim_out = dim_out if dim_out is not None else dim

        act_fn = GELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        self.net.append(act_fn)

        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class AddAuxLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, loss):
        # do nothing in forward (no computation)
        assert loss.numel() == 1
        ctx.requires_aux_loss = loss.requires_grad  
        ctx.dtype = loss.dtype

        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        # add the aux loss gradients
        grad_loss = None
        # put the aux grad the same as the main grad loss
        # aux grad contributes equally
        if ctx.requires_aux_loss:
            grad_loss = torch.ones(1, dtype = ctx.dtype, device = grad_output.device)

        return grad_output, grad_loss
    
class MoEGate(nn.Module):

    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):

        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.alpha = aux_loss_alpha
        self.seq_aux = False

        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        # flatten hidden states
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        # get logits and pass it to softmax
        logits = F.linear(hidden_states, self.weight, bias = None)
        scores = logits.softmax(dim = -1)

        topk_weight, topk_idx = torch.topk(scores, k = self.top_k, dim = -1, sorted = False)

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores

            # used bincount instead of one hot encoding
            counts = torch.bincount(topk_idx.view(-1), minlength = self.n_routed_experts).float()
            ce = counts / topk_idx.numel()  # normalized expert usage

            # mean expert score
            Pi = scores_for_aux.mean(0)

            # expert balance loss
            aux_loss = (Pi * ce * self.n_routed_experts).sum() * self.alpha
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss

class MoEBlock(nn.Module):
    def __init__(self, dim, num_experts: int = 6, moe_top_k: int = 2, dropout: float = 0.0, ff_inner_dim: int = None):
        super().__init__()

        self.moe_top_k = moe_top_k
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            FeedForward(dim, dropout = dropout, inner_dim = ff_inner_dim)
            for _ in range(num_experts)
        ])

        self.gate = MoEGate(dim, num_experts = num_experts, num_experts_per_tok = moe_top_k)
        self.shared_experts = FeedForward(dim, dropout = dropout, inner_dim = ff_inner_dim)

    def forward(self, hidden_states) -> torch.Tensor:

        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        if self.training:

            hidden_states = hidden_states.repeat_interleave(self.moe_top_k, dim = 0)
            y = torch.empty_like(hidden_states, dtype = hidden_states.dtype)

            for i, expert in enumerate(self.experts): 
                tmp = expert(hidden_states[flat_topk_idx == i])
                y[flat_topk_idx == i] = tmp.to(hidden_states.dtype)

            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim = 1)
            y =  y.view(*orig_shape)

            y = AddAuxLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_expert_indices = flat_topk_idx,flat_expert_weights = topk_weight.view(-1, 1)).view(*orig_shape)

        y = y + self.shared_experts(identity)

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()

        # no need for .numpy().cpu() here
        tokens_per_expert = flat_expert_indices.bincount().cumsum(0)
        token_idxs = idxs // self.moe_top_k 
        
        for i, end_idx in enumerate(tokens_per_expert):

            start_idx = 0 if i == 0 else tokens_per_expert[i-1]

            if start_idx == end_idx:
                continue

            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]

            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)

            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 

            # use index_add_ with a 1-D index tensor directly avoids building a large [N, D] index map and extra memcopy required by scatter_reduce_
            # + avoid dtype conversion
            expert_cache.index_add_(0, exp_token_idx, expert_out)

        return expert_cache

def test_moe():

    torch.manual_seed(2025)
    import time

    start = time.time()

    moe_gate = MoEGate(512)
    moe_gate(torch.rand(1, 71, 512))

    moe_block = MoEBlock(512)
    moe_block(torch.rand(1, 77, 512))

    timing = time.time() - start
    print(timing)

if __name__ == "__main__":
    test_moe()