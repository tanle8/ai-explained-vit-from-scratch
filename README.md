# Vision Transformer from Scratch 

A simplified PyTorch implementation of the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).


## Usage

### Install the dependencies
```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py --exp-name vit-with-10-epochs --epochs 10 --batch-size 32
```

The model was trained on the `CIFAR-10` dataset for `67` epochs (hit early stopping) with a batch size of `256` on an apple silicon (M1 pro) (Achieving the similar result when training on Google Colab with CUDA). The learning rate was set to `0.01` and no learning rate schedule was used. The model config was used to train the model:

```python
config = {
    "patch_size": 4,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
}
```

# Tasks
- [x] Upgrade dependencies
- [x] Using W&B
- [x] Refactor and reorganize codebase
- [x] Improve the save checkpoint function to save optimizer states
- [x] Add early stopping
- [x] Training with colab and save results to Google Drive
- [x] Add training with Apple Silicon capability
- [] Experiment with warm-up


## References
- [Vision Transformer from Scratch](https://github.com/tintn/vision-transformer-from-scratch/)
