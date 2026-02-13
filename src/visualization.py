"""
Plotting and visualisation utilities.
"""

import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import PIL
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import CLASS_NAMES


def show_samples(dataset, class_names, n=8):
    """Display random image samples from a dataset.

    Args:
        dataset: A torch Dataset yielding ``(image, label)`` pairs.
        class_names: List of class name strings.
        n: Number of samples to display.
    """
    loader = DataLoader(dataset, batch_size=n, shuffle=True)
    images, labels = next(iter(loader))

    if isinstance(images[0], PIL.Image.Image):
        images = torch.stack([transforms.ToTensor()(img) for img in images])

    plt.figure(figsize=(15, 5))
    for i in range(n):
        img = images[i].permute(1, 2, 0)
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.title(class_names[labels[i]], fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_class_distribution_table(dataset, class_names=CLASS_NAMES):
    """Print class distribution statistics as a formatted table.

    Args:
        dataset: A Flowers102-like dataset with ``._labels``.
        class_names: List of class name strings.
    """
    import pandas as pd

    class_counts = Counter(dataset._labels)
    data = [
        {"Index": idx, "Class Name": class_names[idx], "Count": count}
        for idx, count in sorted(class_counts.items())
    ]
    df = pd.DataFrame(data)
    print(df.to_string(index=False))

    counts = list(class_counts.values())
    print(f"\n- Distribution Summary:")
    print(f"   Mean samples per class: {np.mean(counts):.1f}")
    print(f"   Standard deviation: {np.std(counts):.1f}")
    print(f"   Min samples: {min(counts)}")
    print(f"   Max samples: {max(counts)}")


def explore_text_labels(dataset, n=5):
    """Print random samples to verify dataset labelling.

    Args:
        dataset: A Flowers102-like dataset.
        n: Number of random samples to inspect.
    """
    indices = random.sample(range(len(dataset)), n)
    print("- Random Sample Analysis:")
    for i, idx in enumerate(indices):
        _, label = dataset[idx]
        print(f"   Sample {i + 1} (idx {idx}): Class {label} = '{CLASS_NAMES[label]}'")


def plot_training_curves(train_losses, train_accs, test_accs):
    """Plot training loss and accuracy curves side by side.

    Args:
        train_losses: List of per-epoch training losses.
        train_accs: List of per-epoch training accuracies.
        test_accs: List of per-epoch test/validation accuracies.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, "b-", label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, "b-", label="Training Accuracy")
    ax2.plot(test_accs, "r-", label="Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training vs Test Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_comparison_results(base_acc, novel_acc, base_acc_zs, novel_acc_zs,
                            method_name="CoCoOp"):
    """Compare base vs novel accuracy with zero-shot baseline overlay.

    Args:
        base_acc: Fine-tuned base accuracy (%).
        novel_acc: Fine-tuned novel accuracy (%).
        base_acc_zs: Zero-shot base accuracy (%).
        novel_acc_zs: Zero-shot novel accuracy (%).
        method_name: Display name for the method.
    """
    categories = ["Base Classes", "Novel Classes"]
    accuracies = [base_acc, novel_acc]
    zs_accuracies = [base_acc_zs, novel_acc_zs]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, accuracies, color=["skyblue", "lightcoral"], alpha=0.8)

    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold",
        )

    plt.plot(
        categories, zs_accuracies, color="gray", linestyle="--", marker="o",
        linewidth=2, markersize=8, label="Zero-Shot (CLIP)",
    )

    plt.ylabel("Accuracy (%)")
    plt.title(f"{method_name}: Base vs Novel Class Performance")
    plt.ylim(0, max(accuracies) * 1.2)
    plt.grid(axis="y", alpha=0.3)

    hm = 2 / (1 / base_acc + 1 / novel_acc) if base_acc > 0 and novel_acc > 0 else 0
    plt.text(
        0.5, max(accuracies) * 0.9, f"Harmonic Mean: {hm:.1f}%",
        ha="center", transform=plt.gca().transData, fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classnames, max_classes=20):
    """Plot a confusion matrix for the most frequent classes.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        classnames: List of class name strings.
        max_classes: Number of top classes to display.
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    unique, counts = np.unique(y_true, return_counts=True)
    top_classes = unique[np.argsort(counts)[-max_classes:]]

    mask = np.isin(y_true, top_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes)

    plt.figure(figsize=(12, 10))
    class_labels = [classnames[i] for i in top_classes]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix (Top {max_classes} Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def show_sample_predictions(model, test_loader, device, classnames, num_samples=8):
    """Display sample images with true / predicted labels and confidence.

    Args:
        model: Trained model.
        test_loader: DataLoader for test images.
        device: Torch device.
        classnames: Class name list for display.
        num_samples: Number of samples to show.
    """
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        logits = model(images)
        _, predicted = logits.max(1)
        probs = F.softmax(logits, dim=1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    for i in range(min(num_samples, len(images))):
        img = images[i].cpu() * std + mean
        img = torch.clamp(img, 0, 1)

        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].axis("off")

        true_class = classnames[labels[i].item()]
        pred_class = classnames[predicted[i].item()]
        confidence = probs[i][predicted[i]].item()

        color = "green" if labels[i] == predicted[i] else "red"
        axes[i].set_title(
            f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}",
            fontsize=10, color=color,
        )

    plt.tight_layout()
    plt.show()


def plot_meta_network_analysis(model, test_loader, device, n_samples=100):
    """Visualise the Meta-Network's conditional context generation.

    Args:
        model: A trained CoCoOp model.
        test_loader: DataLoader for analysis.
        device: Torch device.
        n_samples: Maximum number of samples to collect.
    """
    model.eval()

    image_features_list = []
    biases_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            if len(image_features_list) >= n_samples:
                break

            images = images.to(device)
            image_features = model.image_encoder(images.type(model.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            bias = model.prompt_learner.meta_net(image_features)

            image_features_list.append(image_features.cpu().numpy())
            biases_list.append(bias.cpu().numpy())
            labels_list.extend(labels.numpy())

    all_image_features = np.concatenate(image_features_list, axis=0)[:n_samples]
    all_biases = np.concatenate(biases_list, axis=0)[:n_samples]
    all_labels = np.array(labels_list)[:n_samples]

    bias_magnitudes = np.linalg.norm(all_biases, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.hist(bias_magnitudes, bins=30, alpha=0.7, color="purple")
    ax1.set_xlabel("Bias Magnitude")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Meta-Network Bias Magnitudes")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(all_image_features[:, 0], bias_magnitudes,
                alpha=0.6, c=all_labels, cmap="tab10")
    ax2.set_xlabel("First Image Feature Dimension")
    ax2.set_ylabel("Bias Magnitude")
    ax2.set_title("Image Features vs Meta-Network Bias")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"- Meta-Network Analysis:")
    print(f"   Average bias magnitude: {bias_magnitudes.mean():.4f}")
    print(f"   Bias magnitude std: {bias_magnitudes.std():.4f}")
    print(f"   This shows how much the context varies based on input images")
