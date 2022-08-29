import torch 
import haste_pytorch_lib as LIB


if __name__ == "__main__":
    B, T, F, H = 2, 2, 2, 2

    DTYPE = torch.float16

    BATCH_FIRST = True 

    x_ = torch.randn((B, T, F), device="cuda", dtype=DTYPE)
    w = torch.randn((H * 2, F), device="cuda", dtype=DTYPE)
    u = torch.randn((H * 2, H), device="cuda", dtype=DTYPE)
    h_init = torch.randn((B, H), device="cuda", dtype=DTYPE)    
    drop_mask = torch.randn((B, H), device="cuda", dtype=DTYPE) 

    if BATCH_FIRST:
        x = x_.permute(1, 0, 2)


    out, = LIB.ligru_forward(
        True, 
        x.contiguous(),
        h_init,
        w.T.contiguous(),
        u.T.contiguous(),
        drop_mask

    )

    if BATCH_FIRST:
        out = out.permute(1, 0, 2)

    print(x_ @ w.T)
    print(out)
    
    # print(torch.allclose(x_, out))

