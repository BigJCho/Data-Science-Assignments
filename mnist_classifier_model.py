# I still left in unused code
# It should make it easier to spot differences between the two models
# All additions to code are made with respect to processing the MNIST dataset

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# These are the additional imports we now need to process the data
import gzip
import struct
import numpy as np

class LinearClassifier(nn.Module):
    def __init__(self, shape):
        super(LinearClassifier, self).__init__()
        # Arbitrarily selected 64 
        self.fc_the_prequel = nn.Linear(shape, 64)
        # CHANGE SHAPE VARIABLE AS NEEDED TO 64 TO RUN PREQUEL LAYER
        self.fc = nn.Linear(64, 12)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.float()
        x = self.fc_the_prequel(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)

        return x
    
#Process the data
#Process the data
train_image_path = 'train-images-idx3-ubyte.gz'
train_label_path = 'train-labels-idx1-ubyte.gz'
test_image_path = 't10k-images-idx3-ubyte.gz'
test_label_path = 't10k-labels-idx1-ubyte.gz'

# Images
with gzip.open(train_image_path, 'rb') as f:
    # Read over the header
    magic_tri, num_images_tri, rows_tri, cols_tri = struct.unpack(">IIII", f.read(16))
    # Now that we're at the start of the data we read it all in
    image_data_tri = f.read(num_images_tri * rows_tri * cols_tri)
    # Then we convert it to integers
    image_tri = np.frombuffer(image_data_tri, dtype=np.uint8)
    # We now reshape it from a single dimension array to an array of X images each 784 integers long 
    train_image = image_tri.reshape(num_images_tri, rows_tri * cols_tri)
with gzip.open(test_image_path, 'rb') as f:
    # Read over the header
    magic_ti, num_images_ti, rows_ti, cols_ti = struct.unpack(">IIII", f.read(16))
    # Now that we're at the start of the data we read it all in
    image_data_ti = f.read(num_images_ti * rows_ti * cols_ti)
    # Then we convert it to integers
    image_ti = np.frombuffer(image_data_ti, dtype=np.uint8)
    # We now reshape it from a 7840000 array to an array of 10000 images each 784 integers long 
    test_image = image_ti.reshape(num_images_ti, rows_ti * cols_ti)    

# Labels
with gzip.open(train_label_path, 'rb') as f:
    # Read over the header
    magic_trl, num_labels_trl = struct.unpack(">II", f.read(8))
    # Now that we're at the start of the labels we read them all in
    label_data_trl = f.read(num_labels_trl)
    # Then we convert them to integers
    train_labels = np.frombuffer(label_data_trl, dtype=np.uint8)
with gzip.open(test_label_path, 'rb') as f:
    # Read over the header
    magic_tl, num_labels_tl = struct.unpack(">II", f.read(8))
    # Now that we're at the start of the labels we read them all in
    label_data_tl = f.read(num_labels_tl)
    # Then we convert them to integers
    test_labels = np.frombuffer(label_data_tl, dtype=np.uint8)

# Convert to tensors for use in model
train_tensor = torch.tensor(train_image, dtype=torch.float32)
test_tensor = torch.tensor(test_image, dtype=torch.float32)
train_label_tensor = torch.tensor(train_labels, dtype=torch.long)
test_label_tensor = torch.tensor(test_labels, dtype=torch.long)

# MNIST dataset has a shape of 784 for its tensor
shape = 784

### I AM HERE, NEED TO CHANGE THE FORMATS AND MAKE SURE THEY WORK 
# Converts the pandas dataframe into a PyTorch useable format
train_data = TensorDataset(train_tensor, train_label_tensor)
test_data = TensorDataset(test_tensor, test_label_tensor)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Defining what we need for a training loop
model = LinearClassifier(shape)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set to train
model.train()

# Train change epochs as needed
for epoch in range(10):
    for inputs, labels in train_loader:
        inputs = inputs.float()
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Set to eval
model.eval()
# Define trackers for accuracy calculation
correct = 0
total = 0
# Evaluate
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs).squeeze()
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
# Display 
accuracy = correct/total
print(correct)
print(total)
print(f'Test Accuracy: {accuracy * 100:.2f}%') 