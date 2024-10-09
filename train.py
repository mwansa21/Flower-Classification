import argparse
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import os

def load_model(arch, hidden_units):
    """Loads a pre-trained ResNet model and replaces the last layer."""
    if arch == "resnet18":
        # Load a pre-trained resnet18 model
        model = models.resnet18(pretrained=True)
        
        # Freeze all the parameters, so only the final layer is trained
        for param in model.parameters():
            param.requires_grad = False
        
        # Get the number of input features for the final layer
        in_features = model.fc.in_features
        
        # Replace the last layer (fully connected) with our own classifier
        model.fc = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(in_features, hidden_units)),  # First hidden layer
                ("relu1", nn.ReLU()),  # Activation function
                ("dropout1", nn.Dropout(0.5)),  # Dropout to avoid overfitting
                ("fc2", nn.Linear(hidden_units, 102)),  # Final layer with 102 outputs
                ("output", nn.LogSoftmax(dim=1)),  # Log-Softmax for output
            ])
        )
    else:
        raise ValueError("Architecture not supported")
    
    return model

def train(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    """Trains the neural network model."""
    # Check if GPU is available and if the user wants to use it
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    # Define the transformation steps for the training and validation data
    train_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found")
    
    # Load the datasets for training and validation
    train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
    valid_data = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=valid_transforms)
    
    # Create data loaders for training and validation
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=64)

    # Load the model (ResNet18 in this case)
    model = load_model(arch, hidden_units)
    model.to(device)  # Move the model to GPU if available

    # Define the loss function and the optimizer
    criterion = nn.NLLLoss()  # Negative log likelihood loss
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)  # Only update the last layer

    # Training loop
    steps = 0
    running_loss = 0
    print_every = 50  # Print every 50 batches
    
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
            
            optimizer.zero_grad()  # Reset the gradients to zero
            logps = model(inputs)  # Forward pass through the model
            loss = criterion(logps, labels)  # Calculate the loss
            loss.backward()  # Backpropagate the error
            optimizer.step()  # Update the model's weights

            running_loss += loss.item()  # Track the loss
            
            # Every 'print_every' batches, evaluate on the validation set
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()  # Set the model to evaluation mode (turn off dropout)
                
                with torch.no_grad():  # Don't track gradients during validation
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)  # Convert log probabilities back to probabilities
                        top_p, top_class = ps.topk(1, dim=1)  # Get the most likely class
                        equals = top_class == labels.view(*top_class.shape)  # Check if prediction is correct
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Print training and validation results
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                
                running_loss = 0  # Reset running loss after each validation
                model.train()  # Switch back to training mode

    # Save the trained model
    checkpoint = {
        "arch": arch,
        "epoch": epochs,
        "state_dict": model.state_dict(),
        "class_to_idx": train_data.class_to_idx,
        "hidden_units": hidden_units,
    }
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
    print(f"Checkpoint saved to {save_dir}/checkpoint.pth")

# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument("data_dir", help="Directory containing the data")
    parser.add_argument("--save_dir", help="Directory to save checkpoints", default=".")
    parser.add_argument("--arch", help="Architecture of the model", default="resnet18")
    parser.add_argument("--learning_rate", help="Learning rate", type=float, default=0.003)
    parser.add_argument("--hidden_units", help="Number of hidden units", type=int, default=250)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=3)
    parser.add_argument("--gpu", help="Use GPU for training", action="store_true")
    args = parser.parse_args()

    # Call the train function with the arguments
    train(
        args.data_dir,
        args.save_dir,
        args.arch,
        args.learning_rate,
        args.hidden_units,
        args.epochs,
        args.gpu,
    )
