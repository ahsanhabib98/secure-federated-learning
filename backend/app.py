# app.py

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import numpy as np

from flask import Flask, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
import io
from PIL import Image as PILImage
import os
from config import Config
from utils import preprocess_image, generate_hash
from extension import db  # Import db from extensions.py
from model import ImageHash  # Import the model after db is initialized

device = torch.device('cpu')

# Step 2: Define a simple CNN model for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)  
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  
        self.fc1 = nn.Linear(32 * 6 * 6, 64)  
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)  # Initialize db with the app
migrate = Migrate(app, db)

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_PATH'], exist_ok=True)

# Helper Functions
def allowed_file(filename):
    return '.' in filename and \
           os.path.splitext(filename)[1].lower() in app.config['UPLOAD_EXTENSIONS']

def is_image(file):
    try:
        PILImage.open(file)
        file.seek(0)
        return True
    except IOError:
        file.seek(0)
        return False

# Routes
@app.route('/api/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    if file and allowed_file(file.filename) and is_image(file):
        image = PILImage.open(file.stream).convert('RGB')
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Generate hash
        image_hash = generate_hash(image_tensor)

        # Save the hash into the database if it doesn't exist
        existing_hash = ImageHash.query.filter_by(hash=image_hash).first()
        if not existing_hash:
            global_model = SimpleCNN().to(device)
            global_model.load_state_dict(torch.load('global_model.pth'))

            # Disable gradient calculation
            with torch.no_grad():
                output = global_model(image_tensor)  # Output shape: [1, 10]

            # Get the predicted class index
            _, predicted_idx = torch.max(output, 1)
            predicted_class = predicted_idx.item()
            
            new_hash = ImageHash(hash=image_hash, predicted_class=predicted_class)
            db.session.add(new_hash)
            db.session.commit()
            return jsonify({'message': 'New image hash added', 'image_hash': image_hash, 'predicted_class': predicted_class}), 201
        
        return jsonify({'message': 'Already image hash exist', 'image_hash': image_hash, 'predicted_class': existing_hash.predicted_class}), 201

    else:
        return jsonify({'error': 'Invalid image'}), 400

if __name__ == '__main__':
    app.app_context().push()  # Push the app context
    db.create_all()
    app.run(debug=True)
