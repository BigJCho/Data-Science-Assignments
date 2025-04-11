import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader 

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
    
# Process the data
file_path = 'hw6.data.csv.gz'
embeddings = pd.read_csv(file_path, header=None, compression='gzip')

features = embeddings.iloc[:, :-1]
label = embeddings.iloc[:, -1]

# Stats for the data
mean = embeddings.mean()[:-1]
max_value = embeddings.max()[:-1]
stddev = embeddings.std()[:-1]
median = embeddings.median()[:-1]
min_value = embeddings.min()[:-1]

# Normalization and standardization
features_normalized = (features - min_value) / (max_value - min_value)
features_standardized = (features - mean) / stddev

# CHANGE FEATURES VARIABLE AS NEEDED
train_features, test_features, train_labels, test_labels = train_test_split(features, label, test_size=0.3, random_state=42)

# Converts the pandas dataframe into a PyTorch useable format
tr = torch.tensor(train_features.values)
trl = torch.tensor(train_labels.values, dtype= torch.long)
te = torch.tensor(test_features.values)
tel = torch.tensor(test_labels.values, dtype= torch.long)
train_data = TensorDataset(tr, trl)
test_data = TensorDataset(te, tel)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# embeddings are 11 long, but last one is a label, so features are 10 long
shape = 10
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
