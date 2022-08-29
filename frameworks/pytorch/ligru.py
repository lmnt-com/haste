import torch 
import haste_pytorch_lib as LIB

def apply_ligru_cell(x, w, u, h_init, drop_mask):
    wx = x @ w.T 

    hiddens = []
    ht = h_init
    act = torch.nn.ReLU()
    for k in range(wx.shape[1]):
        print(wx[:, k].shape)
        print((ht @ u.T).shape)
        gates = wx[:, k] + ht @ u.T 
        at, zt = gates.chunk(2, 1)
        zt = torch.sigmoid(zt)
        hcand = act(at) * drop_mask
        ht = zt * ht + (1 - zt) * hcand
        hiddens.append(ht)

    h = torch.stack(hiddens, dim=1)
    return h 

if __name__ == "__main__":
    B, T, F, H = 2, 3, 2, 2

    DTYPE = torch.float16

    BATCH_FIRST = True 

    x_ = torch.randn((B, T, F), device="cuda", dtype=DTYPE)
    w = torch.randn((H * 2, F), device="cuda", dtype=DTYPE)
    u = torch.randn((H * 2, H), device="cuda", dtype=DTYPE)
    h_init = torch.randn((B, H), device="cuda", dtype=DTYPE)    
    drop_mask = torch.randn((B, H), device="cuda", dtype=DTYPE) 

    if BATCH_FIRST:
        x = x_.permute(1, 0, 2)


    tmp_wx, tmp_uh, output, cache, = LIB.ligru_forward(
        True, 
        x.clone().contiguous(),
        h_init.clone(),
        w.T.clone().contiguous(),
        u.T.clone().contiguous(),
        drop_mask.clone()

    )

    if BATCH_FIRST:
        tmp_wx = tmp_wx.permute(1, 0, 2)

    h_out_torch = apply_ligru_cell(
        x_, 
        w,
        u,
        h_init,
        drop_mask
    )
    print(h_out_torch)

    print(torch.allclose(tmp_wx, x_ @ w.T))
    
    print(torch.allclose(tmp_uh, h_init @ u.T))

    print(output.permute(1, 0, 2)[:, 1:])
    # print(torch.allclose(x_, out))

