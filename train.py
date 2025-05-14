

import argparse, statistics as stats, torch, math
import torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from torchvision import transforms as T
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)


from models import (create_encoder, GlobalDiscriminator,
                    LocalDiscriminator, PriorDiscriminator)


class Cutout(object):
    def __init__(self, n_holes=1, length=16):
        self.n_holes, self.length = n_holes, length

    def __call__(self, img):
        h, w = img.shape[1:]
        mask = torch.ones_like(img)
        for _ in range(self.n_holes):
            y = torch.randint(0, h, ())
            x = torch.randint(0, w, ())
            y1, y2 = max(0, y-self.length//2), min(h, y+self.length//2)
            x1, x2 = max(0, x-self.length//2), min(w, x+self.length//2)
            mask[:, y1:y2, x1:x2] = 0
        return img * mask


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=.5, beta=1., gamma=.1, eps=1e-6):
        super().__init__()
        self.gD, self.lD, self.pD = GlobalDiscriminator(), LocalDiscriminator(), PriorDiscriminator()
        self.alpha, self.beta, self.gamma, self.eps = alpha, beta, gamma, eps

    def forward(self, y, M, M_):
        y_exp = y.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, M.size(2), M.size(3))
        Ej = -F.softplus(-self.lD(torch.cat((M,  y_exp), 1))).mean()
        Em =  F.softplus( self.lD(torch.cat((M_, y_exp), 1))).mean()
        LOCAL  = (Em - Ej) * self.beta
        Ej = -F.softplus(-self.gD(y, M )).mean()
        Em =  F.softplus( self.gD(y, M_)).mean()
        GLOBAL = (Em - Ej) * self.alpha
        prior = torch.rand_like(y)
        p_real = self.pD(prior).clamp(self.eps, 1-self.eps)
        p_fake = self.pD(y).clamp(self.eps, 1-self.eps)
        PRIOR = -(torch.log(p_real).mean() + torch.log(1 - p_fake).mean()) * self.gamma
        return LOCAL + GLOBAL + PRIOR


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--run_id',     type=int, default=1)
    p.add_argument('--self_attention', action='store_true')
    p.add_argument('--patch_only', action='store_true',
                   help='Use patch-embedding without self-attention')
    p.add_argument('--patch_size', type=int, default=4,
                   help='Size of patch for embedding')
    p.add_argument('--sa_channels', type=int, default=32)
    p.add_argument('--sa_lr',      type=float, default=2.5e-4)
    p.add_argument('--warmup',     type=int, default=5)
    p.add_argument('--resume',     type=int, default=0)
    p.add_argument('--transfer_from', type=int, help="Run ID to transfer from")
    p.add_argument('--se_local', action='store_true',
                   help='Enable channel-attention (SE) on the deep conv map')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    tf_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        Cutout(1, 16)
    ])
    ds = CIFAR10('cifar', train=True, download=True, transform=tf_train)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=4, pin_memory=True)

    
    if args.transfer_from and args.resume > 0:
        print(f"⚡ Transfer learning from run{args.transfer_from} epoch {args.resume}")
        encoder = create_encoder(use_sa=True,
                                 patch_only=args.patch_only,
                                 k_channels=args.sa_channels,
                                 patch_size=args.patch_size,
                                 se_local=args.se_local).to(device)
        transfer_path = Path(f'models/run{args.transfer_from}/encoder{args.resume}.wgt')
        encoder.load_state_dict(torch.load(transfer_path, map_location=device), strict=False)
        for name, param in encoder.named_parameters():
            if 'sa.' not in name:
                param.requires_grad = False
    else:
        encoder = create_encoder(use_sa=args.self_attention,
                                 patch_only=args.patch_only,
                                 k_channels=args.sa_channels,
                                 patch_size=args.patch_size,
                                 se_local=args.se_local).to(device)

    loss_fn = DeepInfoMaxLoss(alpha=0, beta=1.0, gamma=0.1).to(device)

    
    if args.transfer_from:
        sa_params = [p for p in encoder.parameters() if p.requires_grad]
        opt_enc = AdamW(sa_params, lr=args.sa_lr)
    else:
        sa_params = [p for n, p in encoder.named_parameters() if '.sa.' in n]
        base_params = [p for n, p in encoder.named_parameters() if '.sa.' not in n]
        opt_enc = AdamW([
            {'params': base_params, 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': sa_params, 'lr': 0.0 if args.resume==0 else args.sa_lr}
        ])

    opt_loss = AdamW(loss_fn.parameters(), lr=5e-4)
    root = Path(f'models/run{args.run_id}')
    root.mkdir(parents=True, exist_ok=True)

    
    start_epoch = args.resume
    if start_epoch > 0 and not args.transfer_from:
        enc_ckpt = root / f'encoder{start_epoch}.wgt'
        loss_ckpt = root / f'loss{start_epoch}.wgt'
        encoder.load_state_dict(torch.load(enc_ckpt, map_location=device), strict=False)
        loss_fn.load_state_dict(torch.load(loss_ckpt, map_location=device))

        opt_file = root / f'opt{start_epoch}.pt'
        if opt_file.exists():
            try:
                state = torch.load(opt_file, map_location=device)
                if 'opt_enc' in state and 'opt_loss' in state:
                    opt_enc.load_state_dict(state['opt_enc'])
                    opt_loss.load_state_dict(state['opt_loss'])
                else:
                    print(f"Warning: Optimizer state dictionary format not recognized. Starting with fresh optimizer state.")
            except Exception as e:
                print(f"Error loading optimizer state: {e}. Starting with fresh optimizer state.")
        print(f"Resumed from epoch {start_epoch}")

    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if not args.transfer_from:
            if epoch <= args.warmup:
                ramp = epoch / args.warmup
                opt_enc.param_groups[1]['lr'] = ramp * args.sa_lr
            else:
                opt_enc.param_groups[1]['lr'] = args.sa_lr

        pbar, losses = tqdm(loader, ncols=90), []
        for x, _ in pbar:
            x = x.to(device)
            opt_enc.zero_grad()
            opt_loss.zero_grad()

            y, M = encoder(x)
            M_ = torch.cat((M[1:], M[:1]), 0)  
            loss = loss_fn(y, M, M_)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
            opt_enc.step()
            opt_loss.step()

            losses.append(loss.item())
            pbar.set_description(f"Epoch {epoch}  loss {stats.mean(losses[-20:]):.4f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(encoder.state_dict(), root / f'encoder{epoch}.wgt')
            torch.save(loss_fn.state_dict(), root / f'loss{epoch}.wgt')
            torch.save({
                'opt_enc': opt_enc.state_dict(),
                'opt_loss': opt_loss.state_dict()
            }, root / f'opt{epoch}.pt')
            print(f"Saved checkpoints @ epoch {epoch}")


if __name__ == '__main__':
    main()
