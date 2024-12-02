import torch
import torch.nn as nn
import torchvision.transforms as transforms
import hashlib
import pickle
import os

# from federate_learning import SimpleCNN

# # Function to load the global model
# def load_global_model(model_path='global_model.pth'):
#     model = SimpleCNN().cuda()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

# Function to preprocess the input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure the image is the correct size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image)

# Function to generate a hash from the image tensor
def generate_hash(image_tensor):
    image_bytes = image_tensor.cpu().numpy().tobytes()
    return hashlib.sha256(image_bytes).hexdigest()