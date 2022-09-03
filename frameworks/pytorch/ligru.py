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
    def forward(ctx, training, wx, u, h, drop_mask):
        # x = x.permute(1, 0, 2)

        output, cache, = LIB.ligru_forward(
            training, 
            wx.contiguous(),
            h.contiguous(),
            u.T.contiguous(),
            drop_mask.contiguous()

        )
        ctx.save_for_backward(wx, output, u, drop_mask, cache)

        return output
    
    @staticmethod
    def backward(ctx, grad_out):

        wx, h, u, drop_mask, cache, = ctx.saved_tensors

        # wx = wx.permute(2, 0, 1)
        # print(wx.shape)

        # wx = wx.permute(2, 0, 1)
        # print(wx.shape)
        # u = u.permute(1, 0)
        # print(wx.shape)
        du, dwx, dh, = LIB.ligru_backward(
            wx.contiguous(),
            u.contiguous(),
            drop_mask.contiguous(),
            h,
            cache, 
            grad_out.contiguous()
        )


        # print(h.shape)
        # du = dwx.permute(1, 0, 2).T @ h.permute(1, 0, 2) 
        
        # print(dwx.permute(2, 0, 1).T.shape)
        # print(h.shape)
        # print(u.shape)
        # print(dwx.permute(2, 0, 1).T @ h[0])
        # print(dwx.permute(1, 0, 2))
        # print(du.permute(1, 0))

        # print(du.shape)
        # print(dwx.permute(1, 0, 2).shape)
        return None, dwx, du.T, None, None, None 

class VanillaLiGRUCell(torch.autograd.Function):

    relu = torch.nn.ReLU()
    @staticmethod
    def forward(ctx, wx, u, ht, drop_mask):
        hiddens = []
        candidate_gate = []
        update_gate = []
        save_at = []
        
        h_init = ht 
        hiddens.append(h_init)
        # iterate over each timesteps
        for k in range(wx.shape[1]):

            gates = wx[:, k]  + ht @ u.T
            at, zt = gates.chunk(2, 1)
            zt = torch.sigmoid(zt)

            hcand = VanillaLiGRUCell.relu(at) # * drop_mask
            ht = ht * zt + (1 - zt) * hcand

            hiddens.append(ht)
            candidate_gate.append(hcand)
            update_gate.append(zt)
            save_at.append(at)

        # stacks values
        ht = torch.stack(hiddens, dim=1)
        
        zt = torch.stack(update_gate, dim=1)
        at = torch.stack(save_at, dim=1)
        hcand = torch.stack(candidate_gate, dim=1)

        ctx.save_for_backward(h_init, u, wx, zt, at, ht, hcand, drop_mask)
        return ht

    @staticmethod
    def backward(ctx, grad_out):
        h_init, u, wx, zt, at, h, hcand, drop_mask, = ctx.saved_tensors
        
        dzt           = torch.zeros_like(zt)
        dat           = torch.zeros_like(at)
        dh            = torch.zeros_like(h)
        du            = torch.zeros_like(u)
        dwx           = torch.zeros_like(wx)
        dh_prev       = 0

        for t in reversed(range(wx.shape[1])):
            
            ht        = h_init if t - 1 < 0 else h[:, t-1]
            dh        = grad_out[:, t] + dh_prev
            dzt[:, t] = (ht - hcand[:, t]) * dh * (zt[:, t] * (1 - zt[:, t]))
            dat[:, t] = (at[:, t] > 0) * 1.  * 1  * (1 - zt[:, t]) * dh # drop mask
            dwx[:, t] = torch.cat((dat[:,t], dzt[:,t]), 1)
            # print(dwx)
            dh_prev   = zt[:, t] * dh  + dwx[:, t] @ u

        # print(dwx.permute(2, 0, 1).T @ h.permute(2, 0,1 ))
        print(torch.bmm(dwx.permute(2, 0, 1).T , h.permute(2, 0, 1)))
        du  = (dwx.T @ h)
        # print(dwx)
        # print(du)
        # du = dwx.T @ ht
        # print(du.sum(axis=1).shape)
        return dwx, du, dh, None, None

if __name__ == "__main__":
    B, T, F, H = 8, 10, 6, 5

    DTYPE = torch.double

    SEED = 1 
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
    # out.sum().backward()

    print(torch.autograd.gradcheck(ApplyLiGRUCell.apply,
     [True, wx, u_, h_init_, drop_mask_]
    ))
