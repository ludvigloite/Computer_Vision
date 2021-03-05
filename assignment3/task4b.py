import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np



def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


def task4b():
    
    indices = [14, 26, 32, 49, 52]
    
    # Task 4b filter images
    plt.figure(figsize=(15, 6))
    for i in range(len(indices)):
        # Extract and plot filter
        plt.subplot(2,len(indices), i + 1)
        filter = torch_image_to_numpy(first_conv_layer.weight[indices[i],:,:,:])
        plt.imshow(filter)
        plt.title(f"Filter {indices[i]}")

        # Extract and plot corresponding activations
        plt.subplot(2,len(indices),i+len(indices) + 1)
        activation_image = torch_image_to_numpy(activation[0,indices[i],:,:])
        plt.imshow(activation_image, cmap='gray')
        plt.title(f"Activation {indices[i]}")
    
    plt.savefig(f"plots/task4b2.png")
    plt.show()
    
    
    
def task4c(image):
    
    nrFilters = 10
    
    # Forwarding image through the whole network
    image = model.conv1(image)
    image = model.bn1(image)
    image = model.relu(image)
    image = model.maxpool(image)
    image = model.layer1(image)
    image = model.layer2(image)
    image = model.layer3(image)
    image = model.layer4(image)
    
    plt.figure(figsize=(15, 6))
    
    """  # THIS WILL PLOT BOTH ACTIVATION AND FILTER
    
    plt.figure(figsize=(15, 12))
    
    for i in range(5):
        
        # Extract and plot filter
        plt.subplot(4,5, i + 1)
        filter = torch_image_to_numpy(image[0,i,:,:])
        plt.imshow(filter)
        plt.title(f"Filter {i}")
        
        
        # Extract and plot corresponding activations
        plt.subplot(4,5,i+5 + 1)
        activation_image = torch_image_to_numpy(activation[0,i,:,:])
        plt.imshow(activation_image, cmap='gray')
        plt.title(f"Activation {i}")
        
    for j in range(5):
        i = j+10
        
        # Extract and plot filter
        plt.subplot(4,5, i + 1)
        filter = torch_image_to_numpy(image[0,i,:,:])
        plt.imshow(filter)
        plt.title(f"Filter {i}")
        
        
        # Extract and plot corresponding activations
        plt.subplot(4,5,i+5 + 1)
        activation_image = torch_image_to_numpy(activation[0,i,:,:])
        plt.imshow(activation_image, cmap='gray')
        plt.title(f"Activation {i}")
        
    """
    
    for i in range(nrFilters):
        
        # Extract and plot corresponding activations
        plt.subplot(2,5,i + 1)
        activation_image = torch_image_to_numpy(activation[0,i,:,:])
        plt.imshow(activation_image, cmap='gray')
        plt.title(f"Activation {i}")
    
    plt.savefig(f"plots/task4c.png")
    plt.show()
    


if __name__ == "__main__":
    
    image = Image.open("images/zebra.jpg")
    print("Image shape:", image.size)

    model = torchvision.models.resnet18(pretrained=True)
    print(model)
    first_conv_layer = model.conv1
    print("First conv layer weight shape:", first_conv_layer.weight.shape)
    print("First conv layer:", first_conv_layer)

    # Resize, and normalize the image with the mean and standard deviation
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = image_transform(image)[None]
    print("Image shape:", image.shape)

    activation = first_conv_layer(image)
    print("Activation shape:", activation.shape)
    
    #task4b()
    task4c(image)