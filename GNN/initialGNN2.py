# ------------------------------ Modules Import ------------------------------ #
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn import GraphConv
import sklearn.metrics as sm

# ------------------------------ GNN Model Class ----------------------------- #
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree:
        h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function:
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations (Readout):
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

#Create model
classifier_model = Classifier(1, 256, 1)
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier_model.parameters(), lr=0.001)
classifier_model.train()

epoch_losses = []
for epoch in range(10):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(train_data_loader):
        prediction = classifier_model(bg)
        loss = loss_func(torch.sigmoid(prediction), label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)
    
classifier_model.eval()

total = 0
y_pred = []
y_true = []

with torch.no_grad():
    for data in test_data_loader:
        graphs, labels = data
        outputs = torch.softmax(classifier_model(graphs), 1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        y_pred.append(predicted)
        y_true.append(labels)

#Print results
print("Accuracy: %.2f%%" % ((accuracy_score(y_true, y_pred, normalize=False)) / total * 100))
print("Precision: %.2f%%" % (metrics.precision_score(y_true, y_pred) * 100))
print("Recall: %.2f%%" % (metrics.recall_score(y_true, y_pred) * 100))
print("F1-Score: %.2f%%" % (metrics.f1_score(y_true, y_pred) * 100))