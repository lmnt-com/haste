import torch 
import haste_pytorch_lib as LIB



def warmup(fct, *kargs, n_iters=30):
    """Warmup function for JiT."""
    for _ in range(n_iters):
        out = fct(*kargs)
        # out.sum().backward()


def benchmark(fct, *kargs, n_iters=20):
    """Evaluates an input function over n iterations."""
    avg_time = 0
    import time 
    torch.cuda.synchronize()
    for _ in range(n_iters):

        torch.cuda.synchronize()
        time1 = time.time()
        out = fct(*kargs)
        # out.sum().backward()
        torch.cuda.synchronize()
        avg_time += time.time() - time1

    return avg_time / n_iters


class ApplyLiGRUCell(torch.autograd.Function):

    @staticmethod
    def forward(ctx, training, wx, u, h, drop_mask):

        output, cache, act_uh, act_uh_norm_cache, = LIB.ligru_2_0_forward(
            training, 
            wx.contiguous(),
            h.contiguous(),
            u.T.contiguous(),
            drop_mask.contiguous()

        )
        
        ctx.save_for_backward(
            output, 
            cache, 
            act_uh, 
            act_uh_norm_cache,
            wx, 
            u, 
            drop_mask, 
            cache
        )

        return output
    
    @staticmethod
    def backward(ctx, grad_out):

        h, cache, act_uh, act_uh_norm_cache, wx, u, drop_mask, cache, = ctx.saved_tensors

        du, dwx, tmp_dwx, = LIB.ligru_2_0_backward(
            wx.contiguous(),
            u.contiguous(),
            drop_mask.contiguous(),
            h,
            cache, 
            act_uh,
            act_uh_norm_cache,
            grad_out.contiguous()
        )

        # b1 = tmp_dwx.T.unsqueeze(1)
        # r1 = h * b1
        # r1 = torch.sum(r1, dim=-1)
        # print(r1.shape)
        return None, dwx, du.T, None, None, None 


if __name__ == "__main__":
    B, T, F, H = 5, 10, 5, 5
    DTYPE = torch.double

    SEED = 42 
    BATCH_FIRST = True 


    import random
    random.seed(SEED)
    import numpy as np

    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    x_ = torch.randn((B, T, F), device="cuda", dtype=DTYPE)
    w_ = torch.randn((H * 2, F), requires_grad=True, device="cuda", dtype=DTYPE)
    u_ = torch.randn((H * 2, H), requires_grad=True, device="cuda", dtype=DTYPE)
    h_init_ = torch.ones((B, H), requires_grad=False, device="cuda", dtype=DTYPE)    
    drop_mask_ = torch.randn((B, H), device="cuda", dtype=DTYPE) 

    wx = x_ @ w_.T 
    wx = wx.permute(1, 0, 2)

    # out = ApplyLiGRUCell.apply(True, wx, u_, h_init_, drop_mask_)
    # out.permute(1, 0, 2).sum().backward()
    # print(out.permute(1, 0, 2))    
    print(torch.autograd.gradcheck(ApplyLiGRUCell.apply,
     [True, wx, u_, h_init_, drop_mask_]
    ))



    # out = apply_ligru_cell(x_, w_, u_, h_init_, drop_mask_)
    # print(out)
    # warmup(ApplyLiGRUCell.apply, True, wx, u_, h_init_, drop_mask_, n_iters=5)
    # print(benchmark(ApplyLiGRUCell.apply, True, wx, u_, h_init_, drop_mask_, n_iters=10))
