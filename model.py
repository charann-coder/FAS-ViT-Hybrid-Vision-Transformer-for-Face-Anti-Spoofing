import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class XceptionBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.b1 = DepthwiseSeparableConv(in_c, 64)
        self.b2 = DepthwiseSeparableConv(64, 128)
        self.b3 = DepthwiseSeparableConv(128, 128)
    def forward(self, x):
        return self.b3(self.b2(self.b1(x)))

class ASPPMultiScaleBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_c, 64, kernel_size=3, padding=d, dilation=d)
            for d in (1, 2, 3)
        ] + [nn.Conv2d(in_c, 64, 1)])
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.bottle = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        bs = []
        for i, br in enumerate(self.branches):
            if i < 3: bs.append(br(x))
            else:
                tmp = br(F.adaptive_avg_pool2d(x, 1))
                bs.append(tmp.expand_as(bs[0]))
        x = torch.cat(bs, dim=1)
        return self.bottle(self.relu(self.bn(x)))

class LaplacianFrequencyBranch(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.register_buffer('kernel', torch.tensor(
            [[[[0,1,0],[1,-4,1],[0,1,0]]]],
            dtype=torch.float32
        ).repeat(in_c, 1, 1, 1))
        self.conv = nn.Conv2d(in_c, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = F.conv2d(x, self.kernel, padding=1, groups=x.shape[1])
        return self.relu(self.bn(self.conv(x)))

class HybridFASFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.xp = XceptionBlock(3)
        self.aspp = ASPPMultiScaleBlock(3)
        self.lap = LaplacianFrequencyBranch(3)
        self.fuse = nn.Sequential(
            nn.Conv2d(320, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.fuse(torch.cat([self.xp(x), self.aspp(x), self.lap(x)], dim=1))

class SimpleFlashBigBirdAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads, self.scale = heads, dim ** -0.5
        self.qkv = nn.Linear(dim, 3*dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, N, self.heads, C//self.heads).transpose(1,2)
        k = k.view(B, N, self.heads, C//self.heads).transpose(1,2)
        v = v.view(B, N, self.heads, C//self.heads).transpose(1,2)
        attn = (q @ k.transpose(-2,-1)) * self.scale

        # Full attention (dense) — allow tokens to attend to all others
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = attn @ v
        out = out.transpose(1,2).reshape(B,N,C)
        return self.out(out)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimpleFlashBigBirdAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FASViTClassifier(nn.Module):
    def __init__(self, num_classes=2, num_spoof_types=4, img_size=128, patch_size=4, heads=4, depth=2, is_ssl=False, ssl_dim=4):
        super().__init__()
        self.fe = HybridFASFeatureExtractor()
        self.feature_extractor = self.fe  # alias for backward compatibility

        self.patch = nn.Conv2d(192, 192, patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1,1,192))
        self.pos = nn.Parameter(torch.zeros(1, 1+num_patches, 192))
        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(192, heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(192)

        # primary binary head (real/spoof) and separate spoof-type head
        self.head = nn.Linear(192, num_classes)           # real/spoof head (2 classes)
        self.spoof_type_head = nn.Linear(192, num_spoof_types)  # spoof-type head (e.g. 4 types)
    
        self.is_ssl = is_ssl
        if self.is_ssl:
            self.ssl_head = nn.Linear(192, ssl_dim)

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.spoof_type_head.weight, std=0.02)
        nn.init.zeros_(self.spoof_type_head.bias)
        if self.is_ssl:
            nn.init.trunc_normal_(self.ssl_head.weight, std=0.02)
            nn.init.zeros_(self.ssl_head.bias)
    def forward_features(self, x):
        x = self.fe(x)
        x = self.patch(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        B,N,C = x.shape
        cls = self.cls.expand(B,-1,-1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :x.size(1)]
        x = self.transformer(x)
        x = self.norm(x[:,0])   # (B, C)
        if self.is_ssl:
            return self.ssl_head(x) # Only return SSL output
        else:
            real_spoof_out = self.head(x)         # (B, num_classes)
            spoof_type_out = self.spoof_type_head(x)  # (B, num_spoof_types)
            return (real_spoof_out, spoof_type_out)