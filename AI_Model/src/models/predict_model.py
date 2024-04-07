from sklearn.metrics import average_precision_score
import torch
from data.make_dataset import CustomImageSequenceDataset  # Importing CustomImageSequenceDataset from make_dataset module
from models.rally_splitter import ConvRally  # Importing ConvRally model
from torchvision import transforms
from torch.utils.data import DataLoader

# Define transformation for preprocessing images
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create CustomImageSequenceDataset instance for the test dataset
new_test_dataset = CustomImageSequenceDataset('path/to/the/test/dataset', transform=transform, sequence_length=5)

# Define batch size for DataLoader
batch_size = 2

# Create DataLoader instance for iterating over the dataset
test_loader = DataLoader(new_test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the ConvRally model and move it to GPU
model = ConvRally(5).to("cuda")

# Load model checkpoint
checkpoint = torch.load('model_checkpoint5.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Define learning rate and number of epochs
lr = 0.00001
epochs = 1

# Print total number of batches
print("Total batches: ", len(test_loader))

# Set model to training mode
model.train()

# Start training loop
print("Entered into training")
for epoch in range(epochs):
    model.eval()
    loss_cal = 0
    acc = 0
    running_corrects = 0.0
    # Iterate over batches in DataLoader
    for batch_idx, (sequences, targets) in enumerate(test_loader):
        try:
            # Move sequences and targets to GPU
            sequences, targets = sequences.to("cuda"), targets.to("cuda")
            
            # Forward pass
            output_predicted = model(sequences)
            
            # Compute accuracy
            corrects = torch.sum(torch.round(output_predicted) == targets.data)
            running_corrects += corrects
            
            # Convert tensors to numpy arrays for computing average precision
            targets = targets.detach().cpu().numpy()
            output_predicted = output_predicted.detach().cpu().numpy()
            
            # Compute average precision for rally and non-rally classes
            ap_rally = average_precision_score(targets, output_predicted)
            ap_non_rally = average_precision_score(1 - targets, 1 - output_predicted)
            
            # Compute mean average precision (mAP)
            mAP = (ap_rally + ap_non_rally) / 2
            acc += mAP
        except Exception as e:
            print("Error during training:", e)
            break
