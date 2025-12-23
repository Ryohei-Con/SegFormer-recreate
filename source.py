import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


class EfficientAttention(nn.Module):
    def __init__(self, dim, reduction, num_heads=2, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2*dim, bias=False)
        self.scale = (dim // num_heads) ** -0.5
        self.proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.reduction = reduction
        if reduction > 1:
            self.sr = nn.Conv2d(dim, dim, reduction, reduction)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h, w):
        """
        x: (Batch, N, C)
        """
        b, n, c = x.size()

        print(f"x-size at attention\n{x.size()}")
        print(f"{h=}\n{w=}")

        # -> (B, H, N, head_dim)
        # full resolution
        q: torch.Tensor = self.to_q(x).reshape(b, n, self.num_heads, c//self.num_heads).permute(0, 2, 1, 3)

        # -> (B, H, reduced_N, head_dim)
        # spatially reduced
        if self.reduction > 1:
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            x = self.sr(x)
            x = x.permute(0, 2, 3, 1).reshape(b, -1, c)
            x = self.norm(x)
            kv = self.to_kv(x).reshape(b, -1, 2, self.num_heads, c//self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv: torch.Tensor = self.to_kv(x).reshape(b, -1, 2, self.num_heads, c//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # -> (B, H, N, reduced_N)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, -1)
        attn = self.attn_drop(attn)
        # -> (B, H, N, head_dim)
        output = torch.matmul(attn, v)
        output = output.permute(0, 2, 1, 3).reshape(b, n, -1)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output  # (B, N, C)


class MixFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.first_mlp = nn.Linear(dim, 4 * dim)
        self.conv = DWConv(4 * dim)
        self.gelu = nn.GELU()
        self.last_mlp = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor, h, w):
        """Mix FFN

        Args:
            x (torch.Tensor): (Batch, Length, emb_dim)
        """
        b, l, dim = x.size()
        x = self.first_mlp(x)
        x = self.conv(x, h, w)
        x = self.gelu(x)
        x = self.last_mlp(x)
        x = x.reshape(b, l, dim)
        return x  # (Batch, Length, emb_dim)


class DWConv(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.dwConv = nn.Conv2d(hid_dim, hid_dim, kernel_size=3, padding=1, groups=hid_dim)

    def forward(self, x, h, w):
        """
        x: (Batch, Length, emb_dim)
        """
        b, l, emb_dim = x.size()
        x = x.reshape(b, h, w, emb_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.dwConv(x)
        x = x.permute(0, 2, 3, 1)
        return x  # (Batch, Length, emb_dim)


class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channel, out_channel, padding, kernel, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride)
        self.norm = nn.LayerNorm([out_channel])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x (torch.Tensor): (Batch, emb_dim, H, W)
        """
        print(f"x-size at OverlapPatchMerging: {x.size()}")
        x = self.proj(x)  # (Batch, emb_dim, H, W)
        b, d, h, w = x.size()
        x = x.flatten(2).transpose(1, 2)  # (B, N, emb_dim)
        x = self.norm(x)
        return (x, h, w)  # (B, N, emb_dim)


class Block(nn.Module):
    def __init__(self, dim, reduction):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = EfficientAttention(dim, reduction)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, h, w):
        """
        x: (Batch, Length, Emb_dim)
        """
        x = self.dropout(self.attention(self.norm1(x), h, w)) + x
        x_ = self.dropout(self.mlp(self.norm2(x), h, w))
        print(f"x-size at Block: {x.size()}")
        print(f"x_-size at Block: {x_.size()}")
        x = x_ + x
        return x  # (Batch, Length, Emb_dim)


class Encoder(nn.Module):
    def __init__(self, dims, depths, reductions):
        super().__init__()
        self.patch_emb = OverlapPatchMerging(3, dims[0], padding=3, kernel=7, stride=4)

        # stage 1
        self.blocks1 = nn.ModuleList([Block(dims[0], reduction=reductions[0]) for _ in range(depths[0])])
        self.patch_merge1 = OverlapPatchMerging(dims[0], dims[1], padding=1, stride=2, kernel=3)

        # stage 2
        self.blocks2 = nn.ModuleList([Block(dims[1], reduction=reductions[1]) for _ in range(depths[1])])
        self.patch_merge2 = OverlapPatchMerging(dims[1], dims[2], padding=1, stride=2, kernel=3)

        # stage 3
        self.blocks3 = nn.ModuleList([Block(dims[2], reduction=reductions[2]) for _ in range(depths[2])])
        self.patch_merge3 = OverlapPatchMerging(dims[2], dims[3], padding=1, stride=2, kernel=3)

        # stage 4
        self.blocks4 = nn.ModuleList([Block(dims[3], reduction=reductions[3]) for _ in range(depths[3])])
        self.patch_merge4 = OverlapPatchMerging(dims[3], dims[4], padding=1, stride=2, kernel=3)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)
        """
        b, _, _, _ = x.size()
        outputs = []
        x, h, w = self.patch_emb(x)

        for block in self.blocks1:
            x = block(x, h, w)
        b, l, dim = x.size()
        x = x.reshape(b, h, w, dim).permute(0, 3, 1, 2)
        x, h, w = self.patch_merge1(x)
        outputs.append(x.reshape((b, h, w, -1)))

        for block in self.blocks2:
            x = block(x, h, w)
        b, l, dim = x.size()
        x = x.reshape(b, h, w, dim).permute(0, 3, 1, 2)
        x, h, w = self.patch_merge2(x)
        outputs.append(x.reshape((b, h, w, -1)))

        for block in self.blocks3:
            x = block(x, h, w)
        b, l, dim = x.size()
        x = x.reshape(b, h, w, dim).permute(0, 3, 1, 2)
        x, h, w = self.patch_merge3(x)
        outputs.append(x.reshape((b, h, w, -1)))

        for block in self.blocks4:
            x = block(x, h, w)
        b, l, dim = x.size()
        x = x.reshape(b, h, w, dim).permute(0, 3, 1, 2)
        x, h, w = self.patch_merge4(x)
        outputs.append(x.reshape((b, h, w, -1)))

        return outputs


class Decoder(nn.Module):
    def __init__(self, hid_channel, num_classes, dims, image_size: tuple[int, int]):
        super().__init__()
        self.linear1 = nn.Linear(dims[1], hid_channel)
        self.linear2 = nn.Linear(dims[2], hid_channel)
        self.linear3 = nn.Linear(dims[3], hid_channel)
        self.linear4 = nn.Linear(dims[4], hid_channel)

        self.upsample = nn.Upsample(image_size)

        self.all_linear = nn.Linear(4 * hid_channel, hid_channel)
        self.classify = nn.Linear(hid_channel, num_classes)

    def forward(self, outputs):
        # linearに渡すときはCが一番後ろでupsampleに渡すときはCが前
        f1 = self.upsample(self.linear1(outputs[0]).permute(0, 3, 1, 2))
        f2 = self.upsample(self.linear2(outputs[1]).permute(0, 3, 1, 2))
        f3 = self.upsample(self.linear3(outputs[2]).permute(0, 3, 1, 2))
        f4 = self.upsample(self.linear4(outputs[3]).permute(0, 3, 1, 2))
        F = self.all_linear(torch.cat((f1, f2, f3, f4), dim=1).permute(0, 2, 3, 1))
        result = self.classify(F)
        return result


class SegFormer(nn.Module):
    def __init__(self, dims, depths, reductions, hid_channel, num_classes, image_size):
        super().__init__()
        self.encoder = Encoder(dims, depths, reductions)
        self.decoder = Decoder(hid_channel, num_classes, dims, image_size)

    def forward(self, x):
        outputs = self.encoder(x)
        result = self.decoder(outputs)
        return result


class TrainerConfig(BaseModel):
    dims: tuple[int, int, int, int, int]
    depths: tuple[int, int, int, int]
    reductions: tuple[int, int, int, int]
    hid_channel: int
    num_classes: int
    image_size: tuple[int, int]


if __name__ == "__main__":
    x = torch.ones((8, 3, 512, 512))
    config = TrainerConfig(
        dims=(10, 16, 20, 24, 30),
        depths=(2, 2, 2, 2),
        reductions=(64, 16, 4, 1),
        hid_channel=20,
        num_classes=10,
        image_size=(128, 128)
    )
    model = SegFormer(
        config.dims,
        config.depths,
        config.reductions,
        config.hid_channel,
        config.num_classes,
        config.image_size
    )
    pred = model(x)

    print(f"{x.size()=}")
    print(f"{pred.size()=}")
