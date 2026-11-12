"""Reference-conditioned residual U-Net for six 1 km representation-error maps."""
import numpy as np


TARGETS = ["annual_tg", "annual_tn", "winter_tg", "winter_tn", "summer_tg", "summer_tn"]
ARCHITECTURE = "reference-resunet-footprint-v1"


def create_model(torch, width=24, fine=True, pixels=64, spatial_channels=0, pair_channels=25):
    from torch import nn
    import torch.nn.functional as functional
    if width < 8 or width % 8 or pixels < 8 or pixels % 8:
        raise ValueError("Width and output pixels must be positive multiples of eight")
    if pair_channels not in (25, 31):
        raise ValueError("Expected25 relative or31 relative/absolute pair channels")

    class Residual(nn.Module):
        def __init__(self, source, target):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(source, target, 3, padding=1), nn.GroupNorm(8, target), nn.SiLU(),
                nn.Conv2d(target, target, 3, padding=1), nn.GroupNorm(8, target))
            self.skip = nn.Identity() if source == target else nn.Conv2d(source, target, 1)

        def forward(self, x):
            return functional.silu(self.body(x)+self.skip(x))

    class FootprintNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fine = fine
            # 100 m context -> aligned 1 km features; no global spatial pooling here.
            self.stem = nn.Sequential(nn.Conv2d(2, width, 5, stride=5), nn.SiLU(),
                                      nn.Conv2d(width, width, 2, stride=2), nn.SiLU())
            if not fine:
                self.fine_encoder = None
            elif spatial_channels:
                self.fine_encoder = nn.Sequential(
                    nn.Conv2d(spatial_channels, 32, 3, padding=1), nn.SiLU(),
                    nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
            else:
                self.fine_encoder = nn.Sequential(
                    nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.SiLU(),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
                    nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.enc0 = Residual(width+3+(32 if fine else 0), width)
            self.enc1 = Residual(width, width*2)
            self.enc2 = Residual(width*2, width*4)
            self.bottom = Residual(width*4, width*8)
            self.dec2 = Residual(width*12, width*4)
            self.dec1 = Residual(width*6, width*2)
            self.dec0 = Residual(width*3, width)
            self.head = nn.Sequential(nn.Conv2d(width+pair_channels, width, 1), nn.SiLU(),
                                      nn.Conv2d(width, 6, 1), nn.Softplus())
            axis = torch.arange(pixels+16, dtype=torch.float32)-(pixels//2+8)
            yy, xx = torch.meshgrid(-axis, axis, indexing="ij")
            marker = torch.exp(-(xx**2+yy**2)/2)
            self.register_buffer("position", torch.stack([xx/32, yy/32, marker])[None])

        def forward(self, scene, local, features):
            a = self.stem(scene)
            if a.shape[-2:] != self.position.shape[-2:]:
                raise ValueError("Terrain scene has wrong spatial dimensions")
            parts = [a, self.position.expand(len(a), -1, -1, -1)]
            if self.fine:
                encoded = self.fine_encoder(local)
                parts.append(encoded if spatial_channels else
                             encoded[:, :, None, None].expand(-1, -1, *a.shape[-2:]))
            e0 = self.enc0(torch.cat(parts, dim=1))
            e1 = self.enc1(functional.avg_pool2d(e0, 2))
            e2 = self.enc2(functional.avg_pool2d(e1, 2))
            b = self.bottom(functional.avg_pool2d(e2, 2))
            def up(x, skip, layer):
                return layer(torch.cat([functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], dim=1))
            d = up(up(up(b, e2, self.dec2), e1, self.dec1), e0, self.dec0)
            return self.head(torch.cat([d[:, :, 8:-8, 8:-8], features], dim=1))

    return FootprintNet()


def masked_mse(torch, predicted, actual, mask):
    if predicted.shape != actual.shape or mask.shape != actual.shape:
        raise ValueError("Footprint loss shapes differ")
    counts = mask.sum(dim=(0, 2, 3))
    if (counts == 0).any():
        raise ValueError("Every target needs supervised cells in each batch")
    error = torch.where(mask, predicted-actual, torch.zeros_like(predicted))
    return ((error**2).sum(dim=(0, 2, 3))/counts).mean()


def metrics(actual, predicted, mask):
    result = {}
    for i, name in enumerate(TARGETS):
        keep = mask[:, i]
        a, p = actual[:, i][keep].astype(float), predicted[:, i][keep].astype(float)
        if not len(a) or not np.isfinite(a).all() or not np.isfinite(p).all():
            raise ValueError(f"No valid finite scores for {name}")
        error = p-a
        variance = ((a-a.mean())**2).sum()
        result[name] = dict(cells=len(a), rmse=float(np.sqrt(np.mean(error**2))),
                            mae=float(np.mean(abs(error))), bias=float(error.mean()),
                            r2=float(1-(error**2).sum()/variance) if variance > 0 else None)
    return result
