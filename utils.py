import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

from vit import ViTForClassfication


def save_experiment(experiment_name, config, model, optimizer, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    
    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
    
    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)
    
    # Save the model
    save_checkpoint(experiment_name, model, optimizer, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model,optimizer, epoch, base_dir="experiments", save_wandb=False):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)

    # Save model + optimizer states
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, cpfile)

    # Upload the model to checkpoints to W&B
    if save_wandb:
        artifact = wandb.Artifact("model-checkpoints", type="model")
        artifact.add_file(cpfile, name=f"model_{epoch}.pt")
        wandb.log_artifact(artifact)


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)

    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']

    # Load the model
    model = ViTForClassfication(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    
    # Adapt to new checkpoint style
    checkpoint = torch.load(cpfile, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        # New style
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Old style
        model.load_state_dict(checkpoint)

    return config, model, train_losses, test_losses, accuracies


def visualize_images():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Pick 30 samples randomly
    indices = torch.randperm(len(trainset))[:30]
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]
    # Visualize the images using matplotlib
    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])


@torch.no_grad()
def visualize_attention(model, output=None, device=None):
    """
    Visualize the attention maps of the first 30 images from the CIFAR10 test set.
    
    Auto-detects the best device if 'device' is None:
      - 'mps' if available on Apple silicon,
      - else 'cuda' if available,
      - else 'cpu'.
    
    Args:
      model: ViTForClassfication model (or similar) that supports 'output_attentions=True'
      output: Optional path to save the figure (e.g. "attention.png")
      device: 'mps'/'cuda'/'cpu' or None. If None, auto-picks the best device.
    """
    # 1) Auto-pick device if none is provided
    if device is None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print("Using device for attention visualization:", device)

    # 2) Switch model to eval mode
    model.eval()
    
    # 3) Load random images from CIFAR10 test set
    num_images = 30
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Pick 30 samples randomly
    indices = torch.randperm(len(testset))[:num_images]
    raw_images = [np.asarray(testset[i][0]) for i in indices]
    labels = [testset[i][1] for i in indices]
    
    # 4) Convert images to tensors + transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    images = torch.stack([test_transform(img) for img in raw_images])
    
    # 5) Move model & images to device
    images = images.to(device)
    model.to(device)
    
    # 6) Forward pass with 'output_attentions=True'
    logits, attention_maps = model(images, output_attentions=True)
    
    # 7) Predictions
    predictions = torch.argmax(logits, dim=1)
    
    # 8) Concatenate all attention maps from the encoder blocks
    #    shape => (batch_size, total_num_heads, seq_len, seq_len)
    attention_maps = torch.cat(attention_maps, dim=1)
    
    # 9) Select only the attention maps of the CLS token (index 0) attending to the patches (index 1..)
    #    shape => (batch_size, total_num_heads, 1, seq_len-1)
    attention_maps = attention_maps[:, :, 0, 1:]
    
    # 10) Average across heads => shape => (batch_size, seq_len-1)
    attention_maps = attention_maps.mean(dim=1)
    
    # 11) Convert patch dimension to 2D => (batch_size, sqrt(N), sqrt(N))
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    
    # 12) Resize each attention map to (32, 32) via bilinear upsampling
    attention_maps = attention_maps.unsqueeze(1)  # => (batch_size, 1, size, size)
    attention_maps = F.interpolate(
        attention_maps,
        size=(32, 32),
        mode='bilinear',
        align_corners=False
    )
    # => shape => (batch_size, 1, 32, 32)
    attention_maps = attention_maps.squeeze(1)  # => (batch_size, 32, 32)
    
    # 13) Plot each image + attention overlay
    fig = plt.figure(figsize=(20, 10))
    
    # We'll create a mask so that left half is blank
    # The right half overlays the attention map
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
    
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        # Stack the original raw image side-by-side => shape (32, 64, 3)
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        
        # Prepare the attention map for overlay
        # left part => zeros => no overlay
        # right part => attention_map
        extended_attention_map = np.concatenate(
            (np.zeros((32, 32)), attention_maps[i].cpu()), axis=1
        )
        extended_attention_map = np.ma.masked_where(mask == 1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        
        # Show the ground truth + prediction
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        color = "green" if gt == pred else "red"
        ax.set_title(f"gt: {gt} / pred: {pred}", color=color)
    
    # 14) Optionally save figure
    if output is not None:
        plt.savefig(output)
        print("Saved attention visualization to:", output)
    
    plt.show()