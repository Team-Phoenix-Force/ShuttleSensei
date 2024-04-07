import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

class CustomImageSequenceDataset(Dataset):
    def __init__(self, root, train_values_for_video=None, transform=None, target_transform=None, sequence_length=10):
        # Initialize the dataset
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.sequence_length = sequence_length
        self.image_folder = ImageFolder(root, transform=self.transform, target_transform=self.target_transform)
        self.train_values_for_video = train_values_for_video

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.image_folder)

    def __getitem__(self, index):
        # Retrieve a sample from the dataset at the specified index
        sequence = []
        
        num_images = len(self.image_folder)

        # Generate the sequence of images centered around the given index
        for j in range(index - 2*self.sequence_length, index + 2*(self.sequence_length + 1), 5):
            if j < 0 or j >= num_images:
                # Padding for sequences near the start or end
                sequence.append(torch.zeros_like(self.image_folder[0][0]))
            else:
                path, _ = self.image_folder.samples[j]
                image = self.image_folder.loader(path)
                if self.transform:
                    image = self.transform(image)
                sequence.append(image)

        sequence = torch.stack(sequence)
        
        # If train_values_for_video is not provided, return only the image sequence
        if self.train_values_for_video is None:
            return sequence
        
        # Otherwise, return a tuple containing the image sequence and the target value
        target = self.train_values_for_video[index]
        target = target.to(dtype=torch.float32)

        if self.target_transform:
            target = self.target_transform(target)

        return sequence, target
