import torch
import torchvision.transforms as transforms
from PIL import Image

def resize_and_replace(image_path):
    """
    Resize an image to 512x512 using PyTorch and replace the original image.

    Args:
        image_path (str): Path to the input image file.
    """
    # Define transformation to resize image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # Load and transform the original image
    image = Image.open(image_path)
    resized_image = transform(image).unsqueeze(0)

    # Load the replacement image
    replacement_image = Image.open(image_path)
    replacement_tensor = transform(replacement_image).unsqueeze(0)

    # Replace the original image with the resized image
    torch.save(resized_image, image_path)
