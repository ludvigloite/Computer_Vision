import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class Model1(nn.Module):

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
        
        self.num_classes = num_classes
        
        kernel_size_conv = 5
        padding_conv = kernel_size_conv // 2
        print(padding_conv)
        
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        
        self.num_output_features = 4*4*128
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        
        batch_size = x.shape[0]
        feature = self.feature_extractor(x)
        out = self.classifier(feature)
        
        
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out
    
    
    
    
    
class Model2(nn.Module):

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
        
        self.num_classes = num_classes
        
        
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            #using relu as activation func
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        
        self.num_output_features = 4*4*128
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        
        batch_size = x.shape[0]
        feature = self.feature_extractor(x)
        out = self.classifier(feature)
        
        
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    
    modelnr = 1
    
    if modelnr == 1:
        epochs = 10
        batch_size = 64
        learning_rate = 5e-2
        early_stop_count = 4
        dataloaders = load_cifar10(batch_size)
        
        model = Model1(image_channels=3, num_classes=10)
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
        create_plots(trainer, "task3_model1")
        
    else:
        epochs = 10
        batch_size = 64
        learning_rate = 5e-2
        early_stop_count = 4
        dataloaders = load_cifar10(batch_size)
        
        model = Model2(image_channels=3, num_classes=10)
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
        create_plots(trainer, "task3_model2")
