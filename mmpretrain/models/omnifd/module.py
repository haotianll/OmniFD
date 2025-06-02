import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from mmengine.model import BaseModule, ModuleList
from torch import einsum, nn

from mmpretrain.registry import MODELS


def exists(val):
    return val is not None


# REF: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def get_sinusoid_encoding(n_position: int, embed_dims: int):
    vec = torch.arange(embed_dims, dtype=torch.float64)
    vec = (vec - vec % 2) / embed_dims
    vec = torch.pow(10000, -vec).view(1, -1)

    sinusoid_table = torch.arange(n_position).view(-1, 1) * vec
    sinusoid_table[:, 0::2].sin_()
    sinusoid_table[:, 1::2].cos_()

    sinusoid_table = sinusoid_table.to(torch.float32)
    return sinusoid_table.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = rearrange(q, "b t n (h d) -> b h t n d", h=h)
        k = rearrange(k, "b t n (h d) -> b h t n d", h=h)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=h)

        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class InteractionModule(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth=3,
            dim_head=64,
            heads=8,
            num_latents=64,
            ff_mult=4,
    ):
        super().__init__()

        self.dim = dim
        self.num_latents = num_latents

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, T = x.shape[:2]

        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


@MODELS.register_module()
class Interaction(BaseModule):
    def __init__(self,
                 latent_channels=512,
                 feat_channels=None,  # [stage0, stage1, ...]
                 *args,
                 num_tasks=1,
                 init_cfg=dict(type='TruncNormal', layer='Linear', std=0.02),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)

        dim = latent_channels

        if feat_channels is None:
            feat_channels = [dim]

        feat_channels.reverse()

        self.module = InteractionModule(dim=dim, *args, **kwargs)

        self.stage_embed = None

        if len(feat_channels) > 1:
            self.stage_embed = nn.Parameter(torch.randn(len(feat_channels), 1, dim, 1, 1, 1))

        proj_list = []
        for i, channel in enumerate(feat_channels):
            if i == 0 and channel == dim:
                proj = nn.Identity()
            else:
                proj = nn.Sequential(
                    nn.Linear(channel, dim),
                    nn.LayerNorm(dim),
                )
            proj_list.append(proj)
        self.proj = ModuleList(proj_list)

        self.num_tasks = num_tasks

    def forward(self, inputs, data_samples=None, **kwargs):

        masks = None
        if data_samples[0].data_type == 'video':
            if data_samples[0].tasks('video').masks is not None:
                masks = torch.vstack([i.tasks('video').masks for i in data_samples]).unsqueeze(1).float()
            elif data_samples[0].tasks('temporal').masks is not None:
                masks = torch.vstack([i.tasks('temporal').masks for i in data_samples]).unsqueeze(1).float()

        x = []

        for i, (proj, input) in enumerate(zip(self.proj, reversed(inputs))):
            feat = input
            feat = self.proj[i](feat.transpose(1, 4)).transpose(1, 4)

            B, C, T, H, W = feat.shape

            if T > 1:
                if masks is not None:
                    if masks.shape[2] != T:
                        masks = F.interpolate(masks, size=T, mode='nearest')
                    feat = feat * masks.unsqueeze(-1).unsqueeze(-1)

                with torch.no_grad():
                    temporal_embed = get_sinusoid_encoding(T, C)
                    temporal_embed = temporal_embed.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).to(feat.device)
                feat = feat + temporal_embed

            if self.stage_embed is not None:
                feat = feat + self.stage_embed[i]

            feat = feat.contiguous().permute(0, 2, 1, 3, 4).reshape(B, 1, -1, C)
            x.append(feat)

        x = torch.cat(x, dim=-2)

        x = self.module(x)

        x = x.squeeze(1)
        x = x.contiguous().permute(0, 2, 1).contiguous()
        return inputs, x
