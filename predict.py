# Import necessary libraries
import argparse
import json
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict

# Function to load a saved model checkpoint
def load_checkpoint(checkpoint_path):
    # Load the checkpoint file
    checkpoint = torch.load(checkpoint_path)

    # Use a pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)

    # Rebuild the final fully connected layer (fc) of the model to match the checkpoint architecture
    model.fc = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers'])),  # First fully connected layer
            ('relu', nn.ReLU()),  # Activation function
            ('dropout', nn.Dropout(p=0.5)),  # Dropout for regularization
            ('fc2', nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size'])),  # Second fully connected layer
            ('output', nn.LogSoftmax(dim=1))  # Output layer with log softmax activation
        ])
    )

    # Load model state and class-to-index mapping from the checkpoint
    try:
        model.load_state_dict(checkpoint['state_dict'])  # Load the trained parameters
        model.class_to_idx = checkpoint['class_to_idx']  # Load class-to-index dictionary
    except:
        raise ValueError("Something went wrong with loading the checkpoint or model architecture.")

    return model  # Return the loaded model

# Function to preprocess an image for the model
def process_image(image):
    # Define a series of transformations to apply to the image
    preprocess = transforms.Compose([
        transforms.Resize(256),  # Resize image to 256 pixels on the shorter side
        transforms.CenterCrop(224),  # Crop the center 224x224 pixels
        transforms.ToTensor(),  # Convert image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image using ImageNet means
    ])
    
    return preprocess(image).unsqueeze(0)  # Add a batch dimension and return the processed image

# Function to make predictions on an image using the model
def predict(image_path, model, topk=5, gpu=False):
    # Set the device (use GPU if available and requested)
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to the appropriate device
    model.eval()  # Set model to evaluation mode

    # Load and process the image
    image = Image.open(image_path)  # Open the image
    image = process_image(image).to(device)  # Process the image and move it to the device

    # Make predictions
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(image)  # Forward pass through the model
        probs, indices = torch.exp(outputs).topk(topk)  # Get top K probabilities and corresponding indices

        # Convert the results to CPU and numpy arrays
        probs = probs.cpu().numpy().flatten()  # Flatten probabilities
        indices = indices.cpu().numpy().flatten()  # Flatten indices

        # If the model has class_to_idx, map the indices back to the original classes
        if hasattr(model, 'class_to_idx'):
            idx_to_classes = {v: k for k, v in model.class_to_idx.items()}  # Invert the class-to-index dictionary
            indices = [idx_to_classes[idx] for idx in indices]  # Convert indices to original class labels

    return probs, indices  # Return the top probabilities and class labels

# Function to load category names from a JSON file
def load_category_names(json_file):
    # Open the JSON file and load the mapping of category indices to names
    with open(json_file, 'r') as f:
        return json.load(f)

# Main program to handle command-line arguments and run the prediction
if __name__ == "__main__":
    # Create an argument parser to handle command-line input
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained neural network.")
    parser.add_argument('image_path', help="Path to the image file.")  # Image path argument
    parser.add_argument('checkpoint', help="Path to the saved model checkpoint file.")  # Checkpoint path argument
    parser.add_argument('--top_k', type=int, default=3, help="Return top K most likely classes.")  # Optional top K argument
    parser.add_argument('--category_names', help="Path to a JSON file mapping category indices to names.")  # Optional category names file
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference if available.")  # Optional GPU flag

    # Parse the arguments
    args = parser.parse_args()

    # Load the model from the checkpoint
    model = load_checkpoint(args.checkpoint)

    # Make predictions on the input image
    probs, indices = predict(args.image_path, model, args.top_k, args.gpu)

    # If category names are provided, map indices to category names
    if args.category_names:
        category_names = load_category_names(args.category_names)
        output_names = [category_names[str(idx)] for idx in indices]  # Get the category names for the top indices
    else:
        output_names = indices  # Use the class indices directly if no category names are provided

    # Print the top K predictions along with their probabilities
    for i in range(args.top_k):
        print(f"Class: {output_names[i]}, Probability: {probs[i]:.4f}")
