"""
Few-Shot Adaptation of Vision-Language Models using CoOp and CoCoOp.

This package implements prompt learning methods for CLIP-based few-shot
classification on the Oxford Flowers-102 dataset.
"""

from src.constants import CLASS_NAMES
from src.data import get_data, base_novel_categories, split_data, MappedDataset
from src.models import TextEncoder, PromptLearner, CoOp, CoCoOpPromptLearner, CoCoOp
from src.training import train_epoch, evaluate, evaluate_zero_shot, harmonic_mean, main
from src.visualization import (
    show_samples,
    show_class_distribution_table,
    explore_text_labels,
    plot_training_curves,
    plot_comparison_results,
    plot_confusion_matrix,
    show_sample_predictions,
    plot_meta_network_analysis,
)
from src.enhancement import apply_adaptive_contrast_enhancement
from src.analysis import (
    extract_conditional_tokens,
    analyze_token_clusters,
    calculate_advanced_clustering_metrics,
    visualize_token_space,
    analyze_meta_net_tokens_enhanced,
)
