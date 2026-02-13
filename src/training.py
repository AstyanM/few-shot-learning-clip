"""
Training loop, evaluation, and main experiment pipeline.
"""

import clip
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.constants import CLASS_NAMES
from src.data import get_data, base_novel_categories, split_data
from src.models import CoOp, CoCoOp
from src.visualization import (
    plot_training_curves,
    plot_comparison_results,
    plot_confusion_matrix,
    show_sample_predictions,
)


def harmonic_mean(base_accuracy, novel_accuracy):
    """Compute the harmonic mean of base and novel accuracies.

    The harmonic mean penalises imbalanced performance and is the standard
    metric for base-to-novel generalisation evaluation.

    Args:
        base_accuracy: Accuracy on base classes (%).
        novel_accuracy: Accuracy on novel classes (%).

    Returns:
        Harmonic mean value (%).
    """
    if base_accuracy == 0 or novel_accuracy == 0:
        return 0.0
    return 2 / (1 / base_accuracy + 1 / novel_accuracy)


@torch.no_grad()
def evaluate_zero_shot(clip_model, dataset, class_indices, class_names,
                       batch_size=64, device="cuda", label=""):
    """Evaluate CLIP zero-shot classification accuracy.

    Args:
        clip_model: A loaded CLIP model.
        dataset: Dataset to evaluate on.
        class_indices: List of original class indices for text prompts.
        class_names: Full list of class name strings.
        batch_size: Evaluation batch size.
        device: Torch device string.
        label: Description shown in the progress bar.

    Returns:
        Accuracy as a percentage.
    """
    clip_model.eval()

    prompts = [f"a photo of a {class_names[idx]}, a type of flower." for idx in class_indices]
    text_tokens = clip.tokenize(prompts).to(device)

    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc=f"- {label}"):
        images = images.to(device)
        labels = labels.to(device)

        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T
        preds = similarity.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Run a single training epoch.

    Args:
        model: The CoOp or CoCoOp model.
        train_loader: DataLoader for training data.
        optimizer: Optimiser instance.
        device: Torch device.
        epoch: Current epoch number (for display).

    Returns:
        Tuple of ``(average_loss, accuracy_percent)``.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Acc": f"{100. * correct / total:.2f}%",
        })

    pbar.close()
    return total_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, test_loader, device, return_predictions=False):
    """Evaluate a model on a test set.

    Args:
        model: Trained CoOp / CoCoOp model.
        test_loader: DataLoader for evaluation.
        device: Torch device.
        return_predictions: If ``True``, also return per-sample predictions.

    Returns:
        Accuracy (%), or ``(accuracy, preds, labels)`` when
        ``return_predictions=True``.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = logits.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total

    if return_predictions:
        return accuracy, all_preds, all_labels
    return accuracy


def main(model_type="coop", n_ctx=16, ctx_init="a photo of a",
         noise_scale=None, num_epochs=10, batch_size=32, zero_shot_acc=None,
         device=None):
    """Unified training and evaluation pipeline for CoOp / CoCoOp.

    Args:
        model_type: ``"coop"`` or ``"cocoop"``.
        n_ctx: Number of learnable context tokens.
        ctx_init: Initialisation phrase for context vectors.
        noise_scale: Gaussian noise scale for regularisation (``None`` to disable).
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        zero_shot_acc: Pre-computed ``[base_acc, novel_acc]`` to skip re-evaluation.
        device: Torch device. Defaults to CUDA if available.

    Returns:
        Tuple of ``(base_model, novel_model, base_acc, novel_acc,
        zs_base_acc, zs_novel_acc)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    assert model_type in ("coop", "cocoop"), "model_type must be 'coop' or 'cocoop'"

    print(f"- {model_type.upper()} Training on Oxford Flowers 102")
    print("=" * 50)
    print(f"Using device: {device}")

    # Load CLIP
    print("\n- Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/16", device=device)
    if model_type == "cocoop":
        clip_model.float()
    clip_model.eval()

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    writer = SummaryWriter(log_dir=f"runs/{model_type}_{n_ctx}_{ctx_init.replace(' ', '_')}")

    # Datasets
    print("\n- Loading datasets...")
    train_full, val_full, test_full = get_data(transform=preprocess)
    _, _, tmp_test = get_data()
    base_classes, novel_classes = base_novel_categories(tmp_test)

    print(f"\n- Dataset Analysis")
    print(f"Base classes: {len(base_classes)}")
    print(f"Novel classes: {len(novel_classes)}")

    train_base, _ = split_data(train_full, base_classes)
    val_base, val_novel = split_data(val_full, base_classes)
    test_base, test_novel = split_data(test_full, base_classes)

    print(f"\n- Dataset sizes:")
    print(f"Train (base): {len(train_base)}")
    print(f"Val (base): {len(val_base)}, Val (novel): {len(val_novel)}")

    train_loader = DataLoader(train_base, batch_size=batch_size, shuffle=True, num_workers=2)
    val_base_loader = DataLoader(val_base, batch_size=batch_size, shuffle=False, num_workers=2)
    val_novel_loader = DataLoader(val_novel, batch_size=batch_size, shuffle=False, num_workers=2)

    base_classnames = [CLASS_NAMES[i] for i in base_classes]
    novel_classnames = [CLASS_NAMES[i] for i in novel_classes]

    # Zero-shot evaluation
    if not zero_shot_acc:
        print("\n- Evaluating zero-shot accuracy on base and novel classes...")
        zero_shot_base_acc = evaluate_zero_shot(
            clip_model, test_base, base_classes, CLASS_NAMES, device=device, label="Base Zero-Shot",
        )
        zero_shot_novel_acc = evaluate_zero_shot(
            clip_model, test_novel, novel_classes, CLASS_NAMES, device=device, label="Novel Zero-Shot",
        )
    else:
        zero_shot_base_acc, zero_shot_novel_acc = zero_shot_acc

    print(f"Zero-Shot Base Accuracy: {zero_shot_base_acc:.2f}%")
    print(f"Zero-Shot Novel Accuracy: {zero_shot_novel_acc:.2f}%")

    # Model creation
    print(f"\n- Creating {model_type.upper()} model for base classes...")
    ModelClass = CoOp if model_type == "coop" else CoCoOp
    model = ModelClass(base_classnames, clip_model, n_ctx=n_ctx, ctx_init=ctx_init, noise_scale=noise_scale)

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    model.to(device)

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=0.002, momentum=0.9)
    scheduler = None
    if model_type == "coop":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Training
    print(f"\n- Starting {model_type.upper()} training...")
    train_losses, train_accs, val_accs = [], [], []
    best_acc = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        if scheduler:
            scheduler.step()

        if epoch % 5 == 0:
            val_acc = evaluate(model, val_base_loader, device)
            print(f"\nEpoch {epoch}: Train Acc: {train_acc:.2f}%, Val (Base) Acc: {val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"best_{model_type}_model.pth")
        else:
            val_acc = val_accs[-1] if val_accs else 0

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val_base", val_acc, epoch)

    print(f"\n- {model_type.upper()} training completed! Best validation accuracy: {best_acc:.2f}%")

    model.load_state_dict(torch.load(f"best_{model_type}_model.pth"))
    final_base_acc = val_acc if epoch % 5 == 0 else evaluate(model, val_base_loader, device)

    # Novel class transfer
    print(f"\n- Creating {model_type.upper()} model for novel class evaluation...")
    novel_model = ModelClass(novel_classnames, clip_model, n_ctx=n_ctx, ctx_init=ctx_init, noise_scale=noise_scale)

    with torch.no_grad():
        novel_model.prompt_learner.ctx.data = model.prompt_learner.ctx.data.clone()
        if model_type == "cocoop":
            novel_model.prompt_learner.meta_net.load_state_dict(model.prompt_learner.meta_net.state_dict())

    for param in novel_model.parameters():
        param.requires_grad_(False)
    novel_model.to(device)

    print("\n- Evaluating on novel classes...")
    novel_acc = evaluate(novel_model, val_novel_loader, device)

    hm = harmonic_mean(final_base_acc, novel_acc)

    print(f"\n- Final {model_type.upper()} Results:")
    print(f"Base Accuracy:  {final_base_acc:.2f}%")
    print(f"Novel Accuracy: {novel_acc:.2f}%")
    print(f"Harmonic Mean:  {hm:.2f}%")

    # Visualisations
    print("\n- Generating visualizations...")
    plot_training_curves(train_losses, train_accs, val_accs)

    if model_type == "coop":
        print("Noise Scale chosen:", noise_scale)
        novel_acc, novel_preds, novel_labels = evaluate(
            novel_model, val_novel_loader, device, return_predictions=True,
        )
        plot_comparison_results(final_base_acc, novel_acc, zero_shot_base_acc, zero_shot_novel_acc, "CoOp")
        plot_confusion_matrix(novel_labels, novel_preds, novel_classnames, max_classes=20)
        show_sample_predictions(novel_model, val_novel_loader, device, novel_classnames)
    elif model_type == "cocoop":
        plot_comparison_results(final_base_acc, novel_acc, zero_shot_base_acc, zero_shot_novel_acc, "CoCoOp")

    writer.add_hparams(
        {"model_type": model_type, "n_ctx": n_ctx, "ctx_init": ctx_init, "epochs": num_epochs},
        {"hparam/base_acc": final_base_acc, "hparam/novel_acc": novel_acc, "hparam/hmean": hm},
    )
    writer.close()

    return model, novel_model, final_base_acc, novel_acc, zero_shot_base_acc, zero_shot_novel_acc
