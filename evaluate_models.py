# evaluate_models.py  –  full file, patched to handle legacy checkpoints
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from tqdm import tqdm
import statistics as stats
import argparse, glob, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

# ──────────────────── util: add missing attrs on-the-fly ────────────────────
def retrofit_encoder(m):
    """
    Old .mdl files were saved before SE and sa_bn existed.  This adds no-op
    placeholders so they run without retraining.
    """
    import types, torch.nn as nn
    for mod in m.modules():
        name = type(mod).__name__
        if name.startswith("Encoder"):
            if not hasattr(mod, "se"):
                mod.se = nn.Identity()
            if name == "EncoderPatchOnly" and not hasattr(mod, "sa_bn"):
                mod.sa_bn = nn.BatchNorm2d(512)

# ─────────────────── precision / accuracy helper ────────────────────────────
def precision(confusion):
    diag       = confusion * torch.eye(confusion.shape[0], device=confusion.device)
    correct    = diag.sum(0)
    incorrect  = confusion.sum(0) - correct
    prec_class = correct / (correct + incorrect + 1e-12)
    overall    = correct.sum().item() / confusion.sum().item()
    
    # Calculate recall for each class
    recall_class = correct / (confusion.sum(1) + 1e-12)
    
    return prec_class, recall_class, overall

# ─────────────────── find latest model for a run_id ─────────────────────────
def latest_model_path(run_id):
    files = glob.glob(f'./models/run{run_id}/w_dim*.mdl')
    if not files: return None
    return max(files, key=lambda p: int(Path(p).stem.split('w_dim')[-1]))

# ─────────────────── main evaluation routine ────────────────────────────────
def evaluate_model(run_id, device):
    print(f"\nEvaluating run {run_id} …")
    path = latest_model_path(run_id)
    if not path:
        print("  • no .mdl file found")
        return None

    epoch = int(Path(path).stem.split('w_dim')[-1])
    print(f"  • loading {path} (epoch {epoch})")
    try:
        model = torch.load(path, map_location=device)
        retrofit_encoder(model)              # ← patch legacy attrs
        model.to(device).eval()
    except Exception as e:
        print(f"  • load error: {e}")
        return None

    # ---------- data ----------
    ds = CIFAR10('cifar', download=True, transform=ToTensor())
    n_train = len(ds) * 9 // 10
    generator = torch.Generator().manual_seed(42)
    _, test = random_split(ds, [n_train, len(ds) - n_train], generator=generator)
    loader  = DataLoader(test, batch_size=128, shuffle=False)

    crit   = nn.CrossEntropyLoss()
    conf   = torch.zeros(10, 10, device=device)
    losses = []

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"  Run {run_id} Eval", total=len(loader), leave=False):
            x, y = x.to(device), y.to(device)
            out  = model(x)
            losses.append(crit(out, y).item())
            preds = out.argmax(1)
            for p,t in zip(preds, y):
                conf[p, t] += 1

    prec, rec, acc = precision(conf)
    loss_val  = stats.mean(losses)
    print(f"  • test loss {loss_val:.4f}  accuracy {acc:.4f}")

    # save lightweight results dict
    save_dir = Path(f'./models/run{run_id}')
    save_dir.mkdir(parents=True, exist_ok=True)
    results_data = {
        'epoch'      : [epoch],
        'test_loss'  : [loss_val],
        'accuracy'   : [acc],
        'precision'  : prec.cpu(),
        'confusion'  : conf.cpu()
    }
    torch.save(results_data, save_dir / 'results.pt')
    print(f"  • results saved → {save_dir/'results.pt'}")

    return loss_val, acc, prec.cpu().numpy(), rec.cpu().numpy(), conf.cpu().numpy()

# ─────────────────── Plotting Functions ───────────────────────────────────────

def plot_results(results_dict):
    if not results_dict:
        print("\nNo successful evaluations to plot.")
        return

    run_ids = list(results_dict.keys())
    accuracies = [res[1] for res in results_dict.values()]
    all_precisions = [res[2] for res in results_dict.values()] # Collect all precision arrays
    all_recalls = [res[3] for res in results_dict.values()] # Collect all recall arrays

    # Find the run with the highest accuracy (for confusion matrix)
    if not accuracies:
         print("\nNo accuracies found to determine the best run.")
         return
    best_run_index = np.argmax(accuracies)
    best_run_id = run_ids[best_run_index]
    best_accuracy = accuracies[best_run_index]
    best_conf_matrix = results_dict[best_run_id][4]  # Updated index for confusion matrix

    # Define friendly names for runs
    run_name_map = {
        1: 'Original',
        11: 'Patch',
        323: 'SA',
        114: 'SE',
        119: 'Patch 300epochs'
    }
    
    # Get friendly names for x-axis labels
    run_names = [run_name_map.get(r, f'Run {r}') for r in run_ids]
    
    # Get friendly name for best run
    best_run_name = run_name_map.get(best_run_id, f'Run {best_run_id}')
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(class_names)
    num_runs = len(run_ids)

    # Set publication-quality styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Computer Modern Roman'],
        'font.size': 11,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelpad': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Color palette for consistency across plots
    colors = sns.color_palette("muted", num_runs)
    highlight_color = '#e74c3c'  # Red for highlighting the best run
    
    # --- Plot 1: Accuracy per Run ---
    plt.figure(figsize=(8, 6))
    bars = plt.bar(run_names, accuracies, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
    
    # Highlight the best run
    bars[best_run_index].set_color(highlight_color)
    
    plt.title('Overall Accuracy by Model Variant', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Model Variant', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    
    # Set y-axis limits for better visualization
    max_acc = max(accuracies) if accuracies else 0
    min_acc = min(accuracies) if accuracies else 0
    plt.ylim(min_acc - 0.05, max_acc + 0.05)
    
    # Add value labels above bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.005, f"{acc:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.show()

    # --- Plot 2: Grouped Bar Chart for Per-Class Precision Comparison ---
    plt.figure(figsize=(12, 7))
    x = np.arange(num_classes)  # the label locations
    width = 0.8 / num_runs  # Adjust bar width based on number of runs
    offset = (1 - num_runs) * width / 2  # Calculate starting offset

    for i, run_id in enumerate(run_ids):
        precisions = all_precisions[i]
        friendly_name = run_name_map.get(run_id, f'Run {run_id}')
        rects = plt.bar(x + offset + i * width, precisions, width, 
                        label=friendly_name, color=colors[i], 
                        edgecolor='black', linewidth=0.5)

    # Add labels, title and axes ticks
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Per-Class Precision Across Model Variants', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, class_names, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create more visible legend
    plt.legend(title="Model Variants", fontsize=10, 
               title_fontsize=11, bbox_to_anchor=(1.02, 1), 
               loc='upper left', frameon=True, facecolor='white', 
               edgecolor='lightgray')
    
    plt.tight_layout()
    plt.savefig('precision_comparison.png')
    plt.show()
    
    # --- NEW Plot: Grouped Bar Chart for Per-Class Recall Comparison ---
    plt.figure(figsize=(12, 7))
    x = np.arange(num_classes)  # the label locations
    width = 0.8 / num_runs  # Adjust bar width based on number of runs
    offset = (1 - num_runs) * width / 2  # Calculate starting offset

    for i, run_id in enumerate(run_ids):
        recalls = all_recalls[i]
        friendly_name = run_name_map.get(run_id, f'Run {run_id}')
        rects = plt.bar(x + offset + i * width, recalls, width, 
                        label=friendly_name, color=colors[i], 
                        edgecolor='black', linewidth=0.5)

    # Add labels, title and axes ticks
    plt.ylabel('Recall (Class-wise Accuracy)', fontsize=12, fontweight='bold')
    plt.title('Per-Class Recall Across Model Variants', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, class_names, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create more visible legend
    plt.legend(title="Model Variants", fontsize=10, 
               title_fontsize=11, bbox_to_anchor=(1.02, 1), 
               loc='upper left', frameon=True, facecolor='white', 
               edgecolor='lightgray')
    
    plt.tight_layout()
    plt.savefig('recall_comparison.png')
    plt.show()

    # --- Plot 3: Confusion Matrix Heatmap (Best Run) ---
    plt.figure(figsize=(10, 9))
    
    # Calculate normalized confusion matrix for visualization
    norm_conf_matrix = best_conf_matrix / best_conf_matrix.sum(axis=0, keepdims=True)
    
    # Create heatmap with better colormap and annotations
    ax = sns.heatmap(norm_conf_matrix, annot=best_conf_matrix.astype(int), 
                fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, cbar_kws={'label': 'Normalized Frequency'},
                annot_kws={"size": 9})
    
    # Improve colorbar formatting
    cbar = ax.collections[0].colorbar
    cbar.set_label('Normalized Frequency', fontsize=11, fontweight='bold')
    
    plt.title(f'Confusion Matrix for {best_run_name} Model (Accuracy: {best_accuracy:.3f})', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('True Label', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add gridlines to make cells more distinct
    for i in range(num_classes + 1):
        plt.axhline(y=i, color='white', lw=1.5)
        plt.axvline(x=i, color='white', lw=1.5)
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# ─────────────────── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    argp = argparse.ArgumentParser("Evaluate DIM checkpoints and Plot Results")
    argp.add_argument('--runs', nargs='+', type=int, required=True,
                      help='list of run IDs to evaluate, e.g. 1 11 113')
    args = argp.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_results = {}

    print("-" * 60)
    for r in args.runs:
        results = evaluate_model(r, device)
        if results:
            all_results[r] = results
    print("-" * 60)

    plot_results(all_results)