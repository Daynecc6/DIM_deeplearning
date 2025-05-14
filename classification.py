
import torch, torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.optim import Adam
from torchvision.transforms import ToTensor
import statistics as stats
import models
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)


def precision(confusion):
    correct   = confusion * torch.eye(confusion.shape[0], device=confusion.device)
    incorrect = confusion - correct
    correct   = correct.sum(0)
    incorrect = incorrect.sum(0)
    prec_cls  = correct / (correct + incorrect + 1e-12)
    acc       = correct.sum().item() / confusion.sum().item()
    return prec_cls, acc


parser = argparse.ArgumentParser('DeepInfoMax CIFAR-10 evaluation')
parser.add_argument('--run_id',           type=int, default=1)
parser.add_argument('--self_attention',   action='store_true')
parser.add_argument('--sa_channels',      type=int, default=32)
parser.add_argument('--patch_only',       action='store_true')
parser.add_argument('--patch_size',       type=int, default=4)
parser.add_argument('--post_sa',          action='store_true')
parser.add_argument('--encoder_epoch',    type=int, default=100)
parser.add_argument('--epochs',           type=int, default=30)
parser.add_argument('--reload',           type=int, default=None)
parser.add_argument('--fully_supervised', action='store_true')

parser.add_argument('--se_local',         action='store_true',
                    help='Enable Squeeze-Excite in encoder')

args = parser.parse_args()

device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
num_cls    = 10
run_id     = args.run_id

model_dir = Path(f'./models/run{run_id}')
model_dir.mkdir(parents=True, exist_ok=True)


ds = CIFAR10('cifar', download=True, transform=ToTensor())
n_train = len(ds) * 9 // 10
train_ds, test_ds = random_split(ds, [n_train, len(ds) - n_train])
trL = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
teL = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)


classifier, optim, results, start_ep = None, None, None, 1

if args.reload is not None:
    ckpt = model_dir / f'w_dim{args.reload}.mdl'
    if ckpt.exists():
        classifier = torch.load(str(ckpt), map_location=device)
        start_ep   = args.reload + 1
    if classifier:
        optim = Adam(classifier.parameters(), lr=1e-4)
        try:
            results = torch.load(model_dir / 'results.pt')
        except: results = None

if classifier is None:
    if args.fully_supervised:
        encoder = models.create_encoder(
            use_sa     = args.self_attention,
            patch_only = args.patch_only,
            k_channels = args.sa_channels,
            patch_size = args.patch_size,
            se_local   = args.se_local          
        )
        classifier = nn.Sequential(encoder, models.Classifier()).to(device)
    else:
        if args.post_sa:
            classifier = models.PostSaLatent(
                run        = run_id,
                epoch      = args.encoder_epoch,
                use_sa     = args.self_attention,
                patch_only = args.patch_only,
                k_channels = args.sa_channels,
                patch_size = args.patch_size,
                se_local   = args.se_local      
            ).to(device)
        else:
            classifier = models.DeepInfoAsLatent(
                run        = run_id,
                epoch      = args.encoder_epoch,
                use_sa     = args.self_attention,
                patch_only = args.patch_only,
                k_channels = args.sa_channels,
                patch_size = args.patch_size,
                se_local   = args.se_local      
            ).to(device)
    optim   = Adam(classifier.parameters(), lr=1e-4)
    results = {'epoch':[], 'train_loss':[], 'test_loss':[], 'accuracy':[]}

criterion = nn.CrossEntropyLoss()
print(f'Training epochs {start_ep}–{start_ep+args.epochs-1}')


for epoch in range(start_ep, start_ep + args.epochs):
    classifier.train()
    tr_losses = []
    for x,y in tqdm(trL, desc=f'Epoch {epoch} [Train]'):
        x,y = x.to(device), y.to(device)
        optim.zero_grad()
        loss = criterion(classifier(x), y)
        loss.backward()
        optim.step()
        tr_losses.append(loss.item())
    train_loss = stats.mean(tr_losses)

    
    classifier.eval()
    te_losses = []; conf = torch.zeros(num_cls, num_cls, device=device)
    with torch.no_grad():
        for x,y in tqdm(teL, desc=f'Epoch {epoch} [Test ]'):
            x,y = x.to(device), y.to(device)
            preds = classifier(x)
            te_losses.append(criterion(preds,y).item())
            _,p = preds.max(1)
            for pp,tt in zip(p,y): conf[pp,tt] += 1
    test_loss = stats.mean(te_losses)
    prec, acc = precision(conf)

    
    print(f'Epoch {epoch}: TrainLoss={train_loss:.4f}  TestLoss={test_loss:.4f}  Acc={acc:.4f}')
    print(f'Class precisions: {prec}')

    results['epoch'].append(epoch)
    results['train_loss'].append(train_loss)
    results['test_loss'].append(test_loss)
    results['accuracy'].append(acc)

    torch.save(classifier,           model_dir/f'w_dim{epoch}.mdl')
    torch.save(optim.state_dict(),   model_dir/f'opt{epoch}.pt')
    torch.save(results,              model_dir/'results.pt')
    if epoch > start_ep:
        (model_dir/f'opt{epoch-1}.pt').unlink(missing_ok=True)
