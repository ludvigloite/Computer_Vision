import pathlib
import matplotlib.pyplot as plt
import utils
import torchvision
from torch import nn
from dataloaders import load_cifar10, load_cifar10_augmented, load_cifar10_resized
from trainer import Trainer, compute_loss_and_accuracy

class Model(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
                                           # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers
        
        

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        x = self.model(x)
        return x


# -

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
        
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 5
    #dataloaders = load_cifar10(batch_size)
    dataloaders = load_cifar10_resized(batch_size)
    
    model = Model(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    
    trainer.model.eval()
    train_loss, train_acc = compute_loss_and_accuracy(
        trainer.dataloader_train, model, trainer.loss_criterion)
    val_loss, val_acc = compute_loss_and_accuracy(
        trainer.dataloader_val, model, trainer.loss_criterion)
    test_loss, test_acc = compute_loss_and_accuracy(
        trainer.dataloader_test, model, trainer.loss_criterion)
    
    print("\nAccuracies: \n")

    print(f"Train accuracy: \t {train_acc:.3f}")
    print(f"Validation accuracy: \t {val_acc:.3f}")
    print(f"Test accuracy: \t {test_acc:.3f}\n")
    
    #create_plots(trainer, "task4a")
    
    
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)

    plt.figure(figsize=(20, 8))
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"task4a_loss_plot.png"))
