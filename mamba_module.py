import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaAdapter(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        
        # 核心投影层
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 1D 卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # SSM 参数 A 和 D
        # 这里的 d_state 通常是 16
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable Alpha
        self.alpha = nn.Parameter(torch.zeros(1))

    def ssm(self, x):
        """
        运行 SSM (Selective Scan) 的简化 PyTorch 版本
        x: [B, L, D_inner]
        """
        (b, l, d) = x.shape
        
        # 投影获得参数
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*d_state)
        
        # 分割参数
        # dt_rank, d_state=16, d_state=16
        (dt, B, C) = torch.split(x_dbl, [self.dt_rank, 16, 16], dim=-1) 
        dt = F.softplus(self.dt_proj(dt))  # (b, l, d)
        
        A = -torch.exp(self.A_log.float())  # (d, d_state)
        
        # 离散化与循环
        # 这是一个极简循环实现
        y = []
        ht = torch.zeros(b, d, 16, device=x.device) # hidden state
        
        for t in range(l):
            dt_t = dt[:, t, :].unsqueeze(-1) # (b, d, 1)
            A_t = A # (d, n)
            B_t = B[:, t, :].unsqueeze(1) # (b, 1, n)
            C_t = C[:, t, :].unsqueeze(1) # (b, 1, n)
            x_t = x[:, t, :].unsqueeze(-1) # (b, d, 1)
            
            dA = torch.exp(dt_t * A_t)
            dB = dt_t * B_t
            
            ht = dA * ht + dB * x_t
            y_t = (ht * C_t).sum(dim=-1)
            y.append(y_t)
            
        y = torch.stack(y, dim=1) # (b, l, d)
        return y + x * self.D

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        residual = x
        x = self.norm(x)
        
        # Mamba 流程
        # 1. In Proj
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        (x_h, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # 2. Conv1d
        x_h = x_h.transpose(1, 2)
        x_h = self.conv1d(x_h)[:, :, :x.shape[1]]
        x_h = x_h.transpose(1, 2)
        x_h = F.silu(x_h)

        # 3. SSM
        y = self.ssm(x_h)

        # 4. Gating & Out Proj
        y = y * F.silu(res)
        out = self.out_proj(y)
        
        out = self.dropout(out)
        return residual + self.alpha * out
