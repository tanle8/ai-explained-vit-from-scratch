import torch
from torch import nn, optim
import wandb

from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassfication

from config import get_config


config = get_config()

# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device


    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.

        Args:
            trainloader:    DataLoader for training set
            testloader:     DataLoader for validation/test set
            epochs:         number of epochs to run
            save_model_every_n_epochs:  how often to save a checkpoint (0 = never during training)
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        
        # Train the model
        for i in range(epochs):
            # Train for 1 epoch
            train_loss = self.train_epoch(trainloader)

            # Evaluate on test set
            accuracy, test_loss = self.evaluate(testloader)

            # Record stats
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            print(f"Epoch: {i+1}, \
                    Train loss: {train_loss:.4f}, \
                    Test loss: {test_loss:.4f}, \
                    Accuracy: {accuracy:.4f}")
            
            # Log to W&B
            wandb.log({
                "epoch": i+1,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "accuracy": accuracy
            })

            # Periodically save checkpoint (unless it's the very last epoch)
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch ', i+1)
                save_checkpoint(self.exp_name, self.model, i+1, True)
        
        # Finally, save experiment results (model + stats)
        save_experiment(self.exp_name, config, self.model, self.optimizer, train_losses, test_losses, accuracies)


    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.

        Args:
            trainloader:    The DataLoader for the training dataset.

        Returns:
            float:  average loss over the entire training set
        """
        self.model.train()  # set the model in training mode
        total_loss = 0
        
        for batch in trainloader:
            # Move the batch to GPU (if device=cuda)
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            
            # 1) Reset gradient accumulators
            self.optimizer.zero_grad()

            # 2) Forward pass => model(images) returns (logits, attentions)
            #       We only need logits for the actual predictions
            logits = self.model(images)[0]
            
            # 3) Compute the loss
            loss = self.loss_fn(logits, labels)
            
            # 4) Backprop
            loss.backward()

            # 5) Update parameters
            self.optimizer.step()

            # 6) Accumulate total loss => multiply by total number of images (len(images))
            #       because loss.item() is an average over the batch
            total_loss += loss.item() * len(images)

        # Return average loss over the entire dataset
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        """
        Evaluate on test/validation set.

        Returns:
            (accuracy, avg_loss)
        """
        self.model.eval()   # set the model to eval mode
        total_loss = 0
        correct = 0
        
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                
                # 1) Forward => shape (batch_size, num_classes)
                logits, _ = self.model(images)

                # 2) Compute loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        
        return accuracy, avg_loss


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-model-every", type=int, default=0)

    args = parser.parse_args()

    # auto-detect device
    if args.device is None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    return args


def main():
    args = parse_args()
    
    # Training parameters
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    device = args.device
    save_model_every_n_epochs = args.save_model_every

    # Tracking metrics with W&B
    wandb.init(
        project="vit_from_scratch_cifar10",
        name=args.exp_name,
        config=config
    )
    
    # 1) Load the CIFAR10 dataset
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    
    # 2) Create the model, optimizer, loss function
    model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    
    # 3) Create the trainer
    trainer = Trainer(model, optimizer, loss_fn, args.exp_name, device=device)

    # 4) Train the model
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)

    # 5) Finish W&B
    wandb.finish()


if __name__ == "__main__":
    main()