from pathlib import Path


def get_config():
    return {
        "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
        "hidden_size": 48,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 4 * 48,  # 4 * hidden_size
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "image_size": 32,
        "num_classes": 10,  # num_classes of the dataset (CIFAR10)
        "num_channels": 3,
        "qkv_bias": True,
        "use_faster_attention": True,
    }
