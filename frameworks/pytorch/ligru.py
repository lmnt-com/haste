import torch 
import haste_pytorch_lib as LIB

def apply_ligru_cell(x, w, u, h_init, drop_mask):
    wx = x @ w.T 

    hiddens = []
    ht = h_init
    act = torch.nn.ReLU()
    for k in range(wx.shape[1]):
        gates = wx[:, k] + ht @ u.T 
        # print("wx=",wx[:, k])
        # print("uh=",ht @ u.T )
        at, zt = gates.chunk(2, 1)
        # print("at = ", at)
        # print("gates=", gates)

        print("at = ", at)
        zt = torch.sigmoid(zt)

        print("zt = ", zt)
        hcand = act(at) * drop_mask

        print("hcand = ", hcand)
        ht = zt * ht + (1 - zt) * hcand
        hiddens.append(ht)

    h = torch.stack(hiddens, dim=1)
    return h 


class ApplyLiGRUCell(torch.autograd.Function):

    @staticmethod
    def forward(ctx, training, x, w, u, h, drop_mask):

        wx = x @ w.T 
        wx = wx.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        
        output, cache, = LIB.ligru_forward(
            training, 
            wx.contiguous(),
            h.contiguous(),
            u.T.contiguous(),
            drop_mask.contiguous()

        )
        ctx.save_for_backward(wx, h, u, drop_mask, cache)

        return output.permute(1, 0, 2)[:, 1:] 
    
    @staticmethod
    def backward(ctx, grad_out):

        wx, h, u, drop_mask, cache, = ctx.saved_tensors
        
        LIB.ligru_backward(
            wx.T.contiguous(),
            u.T.contiguous(),
            drop_mask,
            h,
            cache, 
            grad_out.contiguous()
        )

        return None, None, None, None, None, None 


if __name__ == "__main__":
    B, T, F, H = 1, 1, 2, 2

    DTYPE = torch.double

    SEED = 12 
    BATCH_FIRST = True 

    import random
    random.seed(SEED)
    import numpy as np

    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    x_ = torch.randn((B, T, F), device="cuda", dtype=DTYPE)
    w_ = torch.randn((H * 2, F), requires_grad=True, device="cuda", dtype=DTYPE)
    u_ = torch.randn((H * 2, H), device="cuda", dtype=DTYPE)
    h_init_ = torch.randn((B, H), device="cuda", dtype=DTYPE)    
    drop_mask_ = torch.randn((B, H), device="cuda", dtype=DTYPE) 

    output = ApplyLiGRUCell.apply(
        True,
        x_,
        w_,
        u_,
        h_init_,
        drop_mask_
    )


    # output.sum().backward()
    

    print("Yes")
    # print(output)


    # torch.autograd.gradcheck(ApplyLiGRUCell.apply,
    #  [True, x_, w_, u_, h_init_, drop_mask_]
    # )

    # import random
    # random.seed(SEED)
    # import numpy as np

    # np.random.seed(SEED)
    # torch.cuda.manual_seed(SEED)
    # torch.manual_seed(SEED)
    # x = torch.randn((B, T, F), device="cuda", dtype=DTYPE)
    # w = torch.randn((H * 2, F), device="cuda", dtype=DTYPE)
    # u = torch.randn((H * 2, H), device="cuda", dtype=DTYPE)
    # h_init = torch.randn((B, H), device="cuda", dtype=DTYPE)    
    # drop_mask = torch.randn((B, H), device="cuda", dtype=DTYPE) 

    # h_out_torch = apply_ligru_cell(
    #     x, 
    #     w,
    #     u,
    #     h_init,
    #     drop_mask
    # )
    # # print(h_out_torch)

    # # print(torch.allclose(tmp_wx, x_ @ w.T))
    
    # # print(torch.allclose(tmp_uh, h_init @ u.T))

    # print(torch.allclose(h_out_torch, output))
    # # print(torch.allclose(x_, out))

