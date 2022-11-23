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

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats // 2)
        self.fc = nn.Linear(h_feats // 2, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        mean_nodes = dgl.mean_nodes(g, 'h')
        graph_rep = self.fc(mean_nodes)
        lsoftmax_vals = F.log_softmax(graph_rep, dim=1)
        return lsoftmax_vals # REVIEW: Could be "F.softmax" or log softmax

def main():
    workspace = os.getcwd()
    datasetDir = f'{workspace}\\Database\\dataset.bin'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Hyper parameters:
    epochs = 200
    batch_size = 1
    n_classes = 2
    lr = 0.005
    
    # Getting the graphs from our database:
    dataset, labels = dgl.load_graphs(datasetDir)
    print(f"First graph's label: {labels['glabel'][0]}") # NOTE: Getting labels.
    # TODO: The graph features should also be in the labels argument.
    print(f"First graph's node features: {dataset[0].nodes()}")
    
    num_examples = len(dataset)
    num_train = int(num_examples * 0.8)
    
    # data split
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))
    
    # data batch for parallel computation
    train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=batch_size, drop_last=False)
    test_dataloader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=batch_size, drop_last=False)
    
    # Output for testing the training dataset:
    # it = iter(train_dataloader)
    # batch = next(it)
    # print(batch)
    
    # batched_graph = batch
    # print('Number of nodes for each graph element in the batch:', batched_graph.batch_num_nodes())
    # print('Number of edges for each graph element in the batch:', batched_graph.batch_num_edges())

    # Recover the original graph elements from the minibatch
    # graphs = dgl.unbatch(batched_graph)
    # print('The original graphs in the minibatch:')
    # print(graphs)
    
    # Create the model with given dimensions
    model = GCN(4, 16, 2)
    # NOTE: 4 = Number of input features, 16 = Number of hidden features, 1 = Number of graph types
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(20):
        for i, batched_graph in enumerate(train_dataloader):
            lsoftmax_vals = model(batched_graph, batched_graph.ndata['feat'].float())
            pred = lsoftmax_vals.argmax(dim=1).float()
            loss = F.binary_cross_entropy(pred[0], labels['glabel'][i].float()) # FIXME: Maybe add more graphs?
            optimizer.zero_grad()
            loss = torch.autograd.Variable(loss, requires_grad = True)
            loss.backward()
            optimizer.step()

    num_correct = 0
    num_tests = 0
    for i, batched_graph in enumerate(test_dataloader):
        pred = model(batched_graph, batched_graph.ndata['feat'].float())
        num_correct += (pred.argmax(1) == labels['glabel'][i]).sum().item()
        num_tests += 1

    print('Test accuracy:', num_correct / num_tests)

# ------------------------------- Run as Script ------------------------------ #
if __name__ == '__main__':
    main()