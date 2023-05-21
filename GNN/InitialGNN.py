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
import matplotlib.pyplot as plt
import seaborn as sns

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats // 2) # TODO - Change the number of hidden layers and features.
        self.fc = nn.Linear(h_feats // 2, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        mean_nodes = dgl.mean_nodes(g, 'h')
        graph_rep = self.fc(mean_nodes)
        # lsoftmax_vals = F.log_softmax(graph_rep, dim=1)
        # return lsoftmax_vals # REVIEW: Could be "F.softmax" or log softmax
        return graph_rep
    
    # def __init__(self, in_dim, hidden_dim, n_classes):
    #     super(GCN, self).__init__()
    #     self.conv1 = GraphConv(in_dim, hidden_dim)
    #     self.conv2 = GraphConv(hidden_dim, hidden_dim)
    #     self.classify = nn.Linear(hidden_dim, n_classes)

    # def forward(self, g, h):
    #     # Perform graph convolution and activation function:
    #     h = F.relu(self.conv1(g, h))
    #     h = F.relu(self.conv2(g, h))
    #     g.ndata['h'] = h
    #     # Calculate graph representation by averaging all the node representations (Readout):
    #     hg = dgl.mean_nodes(g, 'h')
    #     return self.classify(hg)

def main():
    experiment = 2
    
    workspace = os.getcwd()
    datasetDir = f'{workspace}\\Database\\dataset{experiment}.bin'
    # TODO - make sure that same graphs don't get different labels
    
    # TODO - experiment 2 is problematic because the comments are suppose to be labels and not features.
    # TODO - A suggestion: converting all combinations of comments to 1 value with binary operators.
    # TODO - In order to improve experiment 2, try and find the patterns and remove outliers of feedbacks.
    # TODO - experiment 3 - node classification with labels of element score or combination of comments.
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Hyper parameters:
    epochs = 200
    batch_size = 1
    hidden_feats = 16
    n_classes = 2
    lr = 0.005
    
    # TODO - Try experimenting with the threshold
    
    # Getting the graphs from our database:
    dataset, labels = dgl.load_graphs(datasetDir)
    print(f'Dataset size: {len(dataset)}')
    # Getting the number of node features of each graph:
    in_feats = dataset[0].ndata['feat'].size(dim=1)
    
    # Adding self-loops for all graphs:
    for i in range(len(dataset)):
        dataset[i] = dgl.add_self_loop(dataset[i])
    
    # print(f"First graph's label: {labels['glabel'][0]}") # NOTE: Getting labels.
    # TODO: The graph features should also be in the labels argument.
    # print(f"First graph's node features: {dataset[0].nodes()}")
    
    num_examples = len(dataset)
    num_train = int(num_examples * 0.8)
    # TODO - Split the test differently with more even data of pass-fail labels.
    # TODO - Adding 5 fold cross validation
    
    # data split
    train_sampler = torch.arange(num_train)
    test_sampler = torch.arange(num_train, num_examples)
    
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
    model = GCN(in_feats, hidden_feats, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(20):
        for i, batched_graph in enumerate(train_dataloader):
            # TODO - Change the number of graphs in a batch
            lsoftmax_vals = model(batched_graph, batched_graph.ndata['feat'].float())
            pred = lsoftmax_vals.argmax(dim=1).float() #TODO - Put it inside the model.
            loss = F.binary_cross_entropy(pred[0], labels['glabel'][train_sampler[i]].float())
            # TODO - Change loss function to zero-one loss
            loss = torch.autograd.Variable(loss, requires_grad = True) # TODO - Check if this is correct.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    num_correct = 0
    num_tests = 0
    y_true = labels['glabel'][test_sampler].tolist()
    y_pred = []
    for batched_graph in test_dataloader:
        pred = model(batched_graph, batched_graph.ndata['feat'].float())
        num_correct += (pred.argmax(1) == labels['glabel'][test_sampler[num_tests]]).sum().item()
        num_tests += 1
        y_pred.append(pred.argmax(1).tolist())
    accuracy = num_correct / num_tests
    
    # TODO - Check results in comparison with 3 classifiers: all 0, all 1 and random (coin flip) and see if it's better.

    # FIXME: There is a serious problem with the predictions (every run is different)
    print(f'Predictions: {y_pred}')
    print(f'Labels: {y_true}')
    print(f'Accuracy: {sm.accuracy_score(y_true, y_pred)*100:.2f}%')
    print(f'Precision: {sm.average_precision_score(y_true, y_pred)*100:.2f}%')
    print(f'Recall: {sm.recall_score(y_true, y_pred)*100:.2f}%')
    print(f'F1 Score: {sm.f1_score(y_true, y_pred)*100:.2f}%')
    # TODO - Explain why precision is more important in our case and not recall (false positive is more important to reduce)
    
    # ROC curve plot:
    fpr, tpr, _ = sm.roc_curve(y_true,  y_pred)
    roc_auc = sm.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='#607d8b', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='#0091ea', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{workspace}\\GNN\\ROC{experiment}.png', dpi=300)
    plt.clf()
    
    # Confusion matrix plot:
    cm = sm.confusion_matrix(y_true, y_pred)
    # create a heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{workspace}\\GNN\\confusionMatrix{experiment}.png', dpi=300)
    plt.clf()
    
    # Precision-recall curve plot:
    precision, recall, thresholds = sm.precision_recall_curve(y_true, y_pred)
    average_precision = sm.average_precision_score(y_true, y_pred)
    plt.step(recall, precision, color='#0091ea', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='#0091ea')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(f'{workspace}\\GNN\\precisionRecall{experiment}.png', dpi=300)
    plt.clf()

# ------------------------------- Run as Script ------------------------------ #
if __name__ == '__main__':
    main()