import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

# Step 1: Define the CIFAR-10 dataset with basic transformations
def load_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    return trainset, testset

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

# Step 3: Training function for local model
def train_local_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    return model

# Step 4: Aggregate local models (Federated Averaging)
def federated_averaging(global_model, client_models):
    # Averaging model parameters across clients
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([client_models[i].state_dict()[key].float() for i in range(len(client_models))], 0).mean(0)
    
    # Load averaged weights into global model
    global_model.load_state_dict(global_dict)
    return global_model

# Step 5: Test the global model
def test_global_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}%')
    return accuracy

# Step 6: Federated Learning with multiple clients
def federated_learning(num_clients=5, num_rounds=5, epochs=5, batch_size=64):
    # Load CIFAR-10 dataset and split into client datasets
    trainset, testset = load_cifar10(batch_size)
    client_data_len = len(trainset) // num_clients
    client_datasets = random_split(trainset, [client_data_len] * num_clients)
    
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # Initialize global model and client models
    global_model = SimpleCNN().cuda()
    
    for round_num in range(num_rounds):
        print(f"\n### Federated Round {round_num+1} ###")
        client_models = []
        client_loaders = [DataLoader(client_datasets[i], batch_size=batch_size, shuffle=True) for i in range(num_clients)]
        
        # Each client trains its own local model
        for i in range(num_clients):
            start_time = time.time()
            print(f"Client {i+1} training...")
            client_model = SimpleCNN().cuda()  # Each client starts with a new model
            optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            
            client_model = train_local_model(client_model, train_loader=client_loaders[i], criterion=criterion, optimizer=optimizer, epochs=epochs)
            client_models.append(client_model)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken for client {i+1} training: {elapsed_time:.4f} seconds")

        # Aggregate the models using federated averaging
        global_model = federated_averaging(global_model, client_models)
        
        # Test global model after each round
        test_global_model(global_model, test_loader)

# Run federated learning with 5 clients, 5 rounds, and each client training for 5 epochs
federated_learning(num_clients=5, num_rounds=5, epochs=5, batch_size=64)
