from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

# ─────────────────── Channel attention (Squeeze-and-Excite) ───────────────
class SE(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(c, c // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.avg(x))

# ─────────────────── Self-Attention block (pre-norm) ───────────────────────
class ResidualSA(nn.Module):
    def __init__(self, in_channels: int, k_channels: int = 32):
        super().__init__()
        self.norm    = nn.LayerNorm(in_channels)
        self.q = nn.Conv2d(in_channels, k_channels, 1, bias=False)
        self.k = nn.Conv2d(in_channels, k_channels, 1, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.scale   = k_channels ** -0.5
        self.gamma   = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(0.25)
        self.last_attn = None
    def forward(self, x):
        b,c,h,w = x.shape
        y = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        q = self.q(y).reshape(b,-1,h*w).transpose(1,2)
        k = self.k(y).reshape(b,-1,h*w)
        v = self.v(y).reshape(b,-1,h*w)
        attn = (q @ k) * self.scale
        attn = attn.softmax(-1)
        self.last_attn = attn.detach().cpu()
        out = (v @ attn.transpose(1,2)).reshape(b,c,h,w)
        return x + self.dropout(self.gamma * out)

# ───────────────────── Encoders ─────────────────────
class Encoder(nn.Module):
    def __init__(self, se_local=False):
        super().__init__()
        self.c0  = nn.Conv2d(3, 64, 4, 1)
        self.c1, self.bn1 = nn.Conv2d(64,128,4,1), nn.BatchNorm2d(128)
        self.c2, self.bn2 = nn.Conv2d(128,256,4,1), nn.BatchNorm2d(256)
        self.c3, self.bn3 = nn.Conv2d(256,512,4,1), nn.BatchNorm2d(512)
        self.se  = SE(512) if se_local else nn.Identity()
        self.fc  = nn.Linear(512*20*20, 64)
    def forward(self,x):
        h0 = F.relu(self.c0(x))
        f  = F.relu(self.bn1(self.c1(h0)))
        h2 = F.relu(self.bn2(self.c2(f)))
        h2 = F.relu(self.bn3(self.c3(h2)))
        h2 = self.se(h2)
        z  = self.fc(h2.flatten(1))
        return z,f

# ───────────── Patch-only encoder  (BN restored) ─────────────
class EncoderPatchOnly(nn.Module):
    def __init__(self, patch_size=4, se_local=False):
        super().__init__()
        self.c0  = nn.Conv2d(3, 64, 4, 1)
        self.c1, self.bn1 = nn.Conv2d(64,128,4,1), nn.BatchNorm2d(128)
        self.c2, self.bn2 = nn.Conv2d(128,256,4,1), nn.BatchNorm2d(256)
        self.c3, self.bn3 = nn.Conv2d(256,512,4,1), nn.BatchNorm2d(512)
        self.se  = SE(512) if se_local else nn.Identity()

        self.patch = nn.Conv2d(512, 512, patch_size, patch_size)
        self.sa_bn = nn.BatchNorm2d(512)                      # ← restored BN

        g = 20 // patch_size
        self.fc = nn.Linear(512 * g * g, 64)

    def forward(self, x):
        h0 = F.relu(self.c0(x))
        f  = F.relu(self.bn1(self.c1(h0)))
        h2 = F.relu(self.bn2(self.c2(f)))
        h2 = F.relu(self.bn3(self.c3(h2)))
        h2 = self.se(h2)
        hp = self.patch(h2)
        hp = self.sa_bn(hp)                                   # ← BN applied
        z  = self.fc(hp.flatten(1))
        return z, f

class EncoderSA(nn.Module):
    def __init__(self, k_channels=32, patch_size=4, se_local=False):
        super().__init__()
        self.c0  = nn.Conv2d(3,64,4,1)
        self.c1, self.bn1 = nn.Conv2d(64,128,4,1), nn.BatchNorm2d(128)
        self.c2, self.bn2 = nn.Conv2d(128,256,4,1), nn.BatchNorm2d(256)
        self.c3, self.bn3 = nn.Conv2d(256,512,4,1), nn.BatchNorm2d(512)
        self.se  = SE(512) if se_local else nn.Identity()
        self.patch = nn.Conv2d(512,512,patch_size,patch_size)
        self.sa_bn = nn.BatchNorm2d(512)
        self.sa    = ResidualSA(512,k_channels)
        g = 20 // patch_size
        self.fc = nn.Linear(512*g*g,64)
    def forward(self,x):
        h0 = F.relu(self.c0(x))
        f  = F.relu(self.bn1(self.c1(h0)))
        h2 = F.relu(self.bn2(self.c2(f)))
        h2 = F.relu(self.bn3(self.c3(h2)))
        h2 = self.se(h2)
        hp = self.patch(h2)
        hp = self.sa(self.sa_bn(hp))
        z  = self.fc(hp.flatten(1))
        return z,f

# ─────────── factory & rest of file stay unchanged ───────────
def create_encoder(use_sa=False, patch_only=False, k_channels=32, patch_size=4, se_local=False):
    if patch_only:
        return EncoderPatchOnly(patch_size, se_local)
    return EncoderSA(k_channels, patch_size, se_local) if use_sa else Encoder(se_local)

# ────────── Discriminators & Classifier ──────────
class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(128, 64, 3)
        self.c1 = nn.Conv2d( 64, 32, 3)
        self.l0 = nn.Linear(32*22*22 + 64, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512,   1)
    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = F.relu(self.c1(h)).flatten(1)
        h = torch.cat((y, h), dim=1)
        return self.l2(F.relu(self.l1(F.relu(self.l0(h)))))

class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(192, 512, 1)
        self.c1 = nn.Conv2d(512, 512, 1)
        self.c2 = nn.Conv2d(512,   1, 1)
    def forward(self, x):
        return self.c2(F.relu(self.c1(F.relu(self.c0(x)))))

class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(64, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200,    1)
    def forward(self, x):
        return torch.sigmoid(self.l2(F.relu(self.l1(F.relu(self.l0(x))))))

class Classifier(nn.Module):
    def __init__(self, num_classes: int = 10, hidden: int = 200, dropout_p: float = 0.5):
        super().__init__()
        self.fc1     = nn.Linear(64, hidden)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2     = nn.Linear(hidden, num_classes)
        self.l1 = self.fc1
        self.l2 = self.fc2
    def forward(self, pair):
        z, _ = pair
        h = F.relu(self.fc1(z))
        h = self.dropout(h)
        return self.fc2(h)

class DeepInfoAsLatent(nn.Module):
    def __init__(self, run, epoch, use_sa=False, patch_only=False, k_channels=32, patch_size=4, se_local=False):
        super().__init__()
        self.encoder = create_encoder(use_sa, patch_only, k_channels, patch_size, se_local)
        ckpt = Path('models')/f'run{run}'/f'encoder{epoch}.wgt'
        state = torch.load(ckpt, map_location='cpu')
        self.encoder.load_state_dict(state, strict=False)
        self.classifier = Classifier()
    def forward(self, x):
        z, f = self.encoder(x)
        return self.classifier((z.detach(), f))

class PostSaLatent(nn.Module):
    def __init__(self, run, epoch, use_sa=False, patch_only=False, k_channels=32, patch_size=4, se_local=False):
        super().__init__()
        self.encoder = create_encoder(use_sa, patch_only, k_channels, patch_size, se_local)
        ckpt = Path('models')/f'run{run}'/f'encoder{epoch}.wgt'
        state = torch.load(ckpt, map_location='cpu')
        self.encoder.load_state_dict(state, strict=False)
        for p in self.encoder.parameters(): p.requires_grad = False
        self.mha = nn.MultiheadAttention(embed_dim=64, num_heads=k_channels, batch_first=True)
        self.classifier = nn.Linear(64, 10)
    def forward(self, x):
        z, _    = self.encoder(x)
        z_seq,_ = self.mha(z.unsqueeze(1), z.unsqueeze(1), z.unsqueeze(1))
        return self.classifier(z_seq.squeeze(1))
