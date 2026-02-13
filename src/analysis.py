"""
Meta-Net token extraction, clustering analysis, and UMAP visualisation.
"""

import copy

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    pairwise_distances,
    silhouette_samples,
    silhouette_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import CLASS_NAMES
from src.data import base_novel_categories, get_data, split_data
from src.enhancement import apply_adaptive_contrast_enhancement


@torch.no_grad()
def extract_conditional_tokens(meta_net, clip_model, dataset, device,
                               batch_size=32, use_enhancement=True):
    """Extract Meta-Net conditional tokens from a dataset.

    Args:
        meta_net: The Meta-Network (``nn.Sequential``).
        clip_model: A loaded CLIP model (used as the image encoder).
        dataset: Dataset to process.
        device: Torch device.
        batch_size: Batch size for the data loader.
        use_enhancement: Whether to apply adaptive contrast enhancement.

    Returns:
        Dictionary with keys ``conditional_tokens``, ``image_features``,
        ``labels``, ``indices``, ``num_samples``, ``enhancement_used``.
    """
    print(f"Extracting conditional tokens {'with' if use_enhancement else 'without'} enhancement...")
    if use_enhancement:
        print("   - Enhancement method: transformation")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_conditional_tokens = []
    all_image_features = []
    all_labels = []
    all_indices = []

    clip_model.eval()
    meta_net.eval()

    current_idx = 0
    for images, labels in tqdm(dataloader, desc="Processing batches"):
        batch_size_actual = images.size(0)
        images = images.to(device)

        if use_enhancement:
            images = apply_adaptive_contrast_enhancement(images)

        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.float()

        conditional_tokens = meta_net(image_features)

        all_conditional_tokens.append(conditional_tokens.cpu())
        all_image_features.append(image_features.cpu())
        all_labels.extend(labels.tolist())
        all_indices.extend(list(range(current_idx, current_idx + batch_size_actual)))

        current_idx += batch_size_actual

    conditional_tokens = torch.cat(all_conditional_tokens, dim=0)
    image_features = torch.cat(all_image_features, dim=0)

    print(f"- Extracted {conditional_tokens.shape[0]} conditional tokens")
    print(f"- Token dimension: {conditional_tokens.shape[1]}")

    return {
        "conditional_tokens": conditional_tokens,
        "image_features": image_features,
        "labels": all_labels,
        "indices": all_indices,
        "num_samples": len(all_labels),
        "enhancement_used": use_enhancement,
    }


def calculate_advanced_clustering_metrics(token_data, class_names):
    """Compute intra-class compactness, inter-class separation, and discriminability.

    Args:
        token_data: Dictionary returned by :func:`extract_conditional_tokens`.
        class_names: List of class name strings for labelling.

    Returns:
        Dictionary of metric values.
    """
    print("Calculating advanced clustering metrics...")

    tokens = token_data["conditional_tokens"].numpy()
    labels = np.array(token_data["labels"])
    unique_labels = np.unique(labels)

    # Intra-class compactness (lower is better)
    intra_class_distances = []
    class_compactness = {}

    for label in unique_labels:
        mask = labels == label
        class_tokens = tokens[mask]
        if len(class_tokens) > 1:
            class_distances = pairwise_distances(class_tokens, metric="euclidean")
            upper_tri_mask = np.triu(np.ones_like(class_distances, dtype=bool), k=1)
            class_intra_distances = class_distances[upper_tri_mask]
            intra_class_distances.extend(class_intra_distances)
            class_compactness[class_names[label]] = np.mean(class_intra_distances)
        else:
            class_compactness[class_names[label]] = 0.0

    overall_intra_compactness = np.mean(intra_class_distances) if intra_class_distances else 0.0

    # Inter-class separation (higher is better)
    class_centroids = {}
    for label in unique_labels:
        class_centroids[label] = np.mean(tokens[labels == label], axis=0)

    inter_class_distances = []
    centroid_labels = list(class_centroids.keys())
    for i in range(len(centroid_labels)):
        for j in range(i + 1, len(centroid_labels)):
            dist = np.linalg.norm(class_centroids[centroid_labels[i]] - class_centroids[centroid_labels[j]])
            inter_class_distances.append(dist)

    overall_inter_separation = np.mean(inter_class_distances) if inter_class_distances else 0.0
    min_inter_separation = np.min(inter_class_distances) if inter_class_distances else 0.0

    discriminability_ratio = overall_inter_separation / (overall_intra_compactness + 1e-8)

    # Per-class silhouette
    silhouette_scores = silhouette_samples(tokens, labels)
    class_silhouette = {
        class_names[label]: np.mean(silhouette_scores[labels == label])
        for label in unique_labels
    }

    print(f"- Advanced Clustering Metrics:")
    print(f"   Intra-class Compactness: {overall_intra_compactness:.4f}")
    print(f"   Inter-class Separation: {overall_inter_separation:.4f}")
    print(f"   Min Inter-class Distance: {min_inter_separation:.4f}")
    print(f"   Discriminability Ratio: {discriminability_ratio:.4f}")

    return {
        "intra_class_compactness": overall_intra_compactness,
        "inter_class_separation": overall_inter_separation,
        "min_inter_separation": min_inter_separation,
        "discriminability_ratio": discriminability_ratio,
        "class_compactness": class_compactness,
        "class_silhouette": class_silhouette,
    }


def analyze_token_clusters(token_data, class_names, n_clusters=None):
    """Run K-means clustering and compute quality metrics on conditional tokens.

    Args:
        token_data: Dictionary returned by :func:`extract_conditional_tokens`.
        class_names: List of class name strings.
        n_clusters: Number of clusters (defaults to number of unique labels).

    Returns:
        Dictionary with clustering results and metrics.
    """
    print("Analyzing token clustering patterns...")

    tokens = token_data["conditional_tokens"].numpy()
    true_labels = np.array(token_data["labels"])

    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tokens)

    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    silhouette_avg = silhouette_score(tokens, cluster_labels)

    advanced_metrics = calculate_advanced_clustering_metrics(token_data, class_names)

    print(f"- Basic Clustering Results:")
    print(f"   Number of clusters: {n_clusters}")
    print(f"   Adjusted Rand Index: {ari_score:.3f}")
    print(f"   Silhouette Score: {silhouette_avg:.3f}")

    cluster_purity = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]
        if len(cluster_true_labels) > 0:
            most_common_label = np.bincount(cluster_true_labels).argmax()
            purity = np.sum(cluster_true_labels == most_common_label) / len(cluster_true_labels)
            cluster_purity.append(purity)

    overall_purity = np.mean(cluster_purity)
    print(f"   Overall Cluster Purity: {overall_purity:.3f}")

    return {
        "cluster_labels": cluster_labels,
        "ari_score": ari_score,
        "silhouette_score": silhouette_avg,
        "cluster_purity": overall_purity,
        "advanced_metrics": advanced_metrics,
    }


def visualize_token_space(token_data, max_samples_per_class=5):
    """Reduce conditional tokens to 2-D with UMAP and plot by class.

    Args:
        token_data: Dictionary returned by :func:`extract_conditional_tokens`.
        max_samples_per_class: Cap per class for cleaner plots.

    Returns:
        Array of 2-D reduced coordinates.
    """
    print("Visualizing token space using UMAP...")

    tokens = token_data["conditional_tokens"].numpy()
    labels = np.array(token_data["labels"])

    if max_samples_per_class:
        sampled_indices = []
        for label in np.unique(labels):
            label_indices = np.where(labels == label)[0]
            if len(label_indices) > max_samples_per_class:
                sampled = np.random.choice(label_indices, max_samples_per_class, replace=False)
                sampled_indices.extend(sampled)
            else:
                sampled_indices.extend(label_indices)

        sampled_indices = np.array(sampled_indices)
        tokens = tokens[sampled_indices]
        labels = labels[sampled_indices]
        print(f"- Sampled {len(tokens)} tokens for visualization")

    reducer = umap.UMAP(n_components=2, random_state=42)
    print("- Applying UMAP reduction...")
    reduced_tokens = reducer.fit_transform(tokens)

    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            reduced_tokens[mask, 0], reduced_tokens[mask, 1],
            c=[colors[i]], label=CLASS_NAMES[label], alpha=0.7, s=30,
        )

    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.show()

    return reduced_tokens


def analyze_meta_net_tokens_enhanced(base_model, dataset_type="train",
                                     batch_size=32, device=None):
    """Full pipeline: extract, cluster, and visualise Meta-Net tokens.

    Runs both baseline (no enhancement) and enhanced (contrast-boosted)
    analyses and prints a comparative summary table.

    Args:
        base_model: Trained CoCoOp model to extract the Meta-Net from.
        dataset_type: One of ``"train"``, ``"val"``, ``"test_base"``, ``"test_novel"``.
        batch_size: Batch size for token extraction.
        device: Torch device. Defaults to CUDA if available.

    Returns:
        Dictionary with ``"baseline"`` and ``"enhanced_method"`` results.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Meta-Net Tokens Visualization and Analysis")
    print("=" * 80)

    print("\n1. Loading Meta-Net...")
    meta_net = copy.deepcopy(base_model.prompt_learner.meta_net)

    print("\n2. Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_model.float()
    print("- CLIP ViT-B/16 loaded successfully")

    print(f"\n3. Loading {dataset_type} dataset...")
    train_set, val_set, test_set = get_data(transform=preprocess)
    base_classes, novel_classes = base_novel_categories(train_set)

    if dataset_type == "train":
        dataset, _ = split_data(train_set, base_classes)
        class_names = [CLASS_NAMES[c] for c in base_classes]
    elif dataset_type == "val":
        dataset, _ = split_data(val_set, base_classes)
        class_names = [CLASS_NAMES[c] for c in base_classes]
    elif dataset_type == "test_base":
        dataset, _ = split_data(test_set, base_classes)
        class_names = [CLASS_NAMES[c] for c in base_classes]
    elif dataset_type == "test_novel":
        _, dataset = split_data(test_set, base_classes)
        class_names = [CLASS_NAMES[c] for c in novel_classes]
    else:
        raise ValueError("dataset_type must be one of: 'train', 'val', 'test_base', 'test_novel'")

    print(f"- Dataset loaded: {len(dataset)} samples")
    print(f"- Classes: {len(class_names)}")

    results = {}

    # Baseline (no enhancement)
    print("\n4. Baseline Analysis (No Enhancement)...")
    baseline_tokens = extract_conditional_tokens(
        meta_net, clip_model, dataset, device, batch_size, use_enhancement=False,
    )
    baseline_clusters = analyze_token_clusters(baseline_tokens, class_names)
    results["baseline"] = {"token_data": baseline_tokens, "cluster_results": baseline_clusters}
    visualize_token_space(baseline_tokens)

    # Enhanced
    print(f"\n5. Enhanced Analysis...")
    enhanced_tokens = extract_conditional_tokens(
        meta_net, clip_model, dataset, device, batch_size, use_enhancement=True,
    )
    enhanced_clusters = analyze_token_clusters(enhanced_tokens, class_names)
    results["enhanced_method"] = {"token_data": enhanced_tokens, "cluster_results": enhanced_clusters}
    visualize_token_space(enhanced_tokens)

    # Summary table
    print("\n" + "=" * 80)
    print("Comparative Analysis Summary")
    print("=" * 80)

    summary_table = []
    for key, result in results.items():
        metrics = result["cluster_results"]
        adv = metrics["advanced_metrics"]
        summary_table.append({
            "Method": key,
            "ARI": f"{metrics['ari_score']:.3f}",
            "Silhouette": f"{metrics['silhouette_score']:.3f}",
            "Purity": f"{metrics['cluster_purity']:.3f}",
            "Discriminability": f"{adv['discriminability_ratio']:.3f}",
            "Compactness": f"{adv['intra_class_compactness']:.3f}",
            "Separation": f"{adv['inter_class_separation']:.3f}",
        })

    print(f"{'Method':<15} {'ARI':<6} {'Sil':<6} {'Pur':<6} {'Disc':<6} {'Comp':<6} {'Sep':<6}")
    print("-" * 60)
    for row in summary_table:
        print(
            f"{row['Method']:<15} {row['ARI']:<6} {row['Silhouette']:<6} "
            f"{row['Purity']:<6} {row['Discriminability']:<6} "
            f"{row['Compactness']:<6} {row['Separation']:<6}"
        )
    print("=" * 80)

    return results
