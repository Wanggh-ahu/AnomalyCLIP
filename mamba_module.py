import torch
import torch.nn as nn

class MambaAdapter(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 尝试导入官方 Mamba
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
        except ImportError:
            print("Warning: mamba_ssm not found. Using a simple Mamba-like Identity block for debugging.")
            # 这是一个简单的占位符，保证代码能跑，但没有 Mamba 效果
            # 建议安装 mamba-ssm 或者找纯 PyTorch 实现替换这里
            self.mamba = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model)
            )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 这里的 alpha 初始为 0，让训练初期更稳定
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + self.alpha * x
