# ------------------------------ Modules Imports ----------------------------- #
# Official modules:
import os
import timeit
import csv
import pandas as pd
import dgl
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# AIPDORCS modules:
from classes import *

# ------------------------- Get Elements Information ------------------------- #
def getElementsInfo(fileName: str, 
                    timeDebug:bool=False):
    """Extracting structural elements information to a list from a given filename.
    
    Parameters
    ----------
        ``fileName (str)``: A path to a file.
        ``timeDebug (bool, optional)``: Whether or not to print function timing for debug, False by default.
    
    Returns
    -------
        ``(list[Element])``: A list of all structural elements information.
    
    """    
    startTime = timeit.default_timer()
    # Dictionary of number of features for each element type:
    featuresDict = Element.featuresDict
    
    # Lists of elements information:
    elemIDs = []
    elemConnections = []
    elemGeoFeatures = []
    elemList = []
    tempList = []
    
    # Getting the number of geometric features in the given file:
    elementType = [item for item in featuresDict if item in fileName]
    featureCount = featuresDict[elementType[0]]
    
    # Getting element information from the database:
    with open(fileName, 'r') as csvFile:
        elementsFile = csv.reader(csvFile, delimiter=',')
        # Loop over lines in the file:
        for line in elementsFile:
            match line[0]:
                case 'Beam ID' | 'Column ID' | 'Slab ID' | 'Wall ID':
                    elemIDs.append(line[1])
                
                case 'Beam Connections' | 'Column Connections' | \
                    'Slab Connections' | 'Wall Connections':
                    # Removing duplicates (if exist):
                    connections = list(set(line[1:]))
                    connections.sort()
                    elemConnections.append(connections)
                
                case other: # other = geometric features
                    tempList.append(line[1])
                    # Adding all elements' geometric features together:
                    if len(tempList) == featureCount:
                        elemGeoFeatures.append(tempList)
                        tempList = []
    
    # Creating a list of all elements with all their information from the CSV files:
    for i in range(len(elemIDs)):
        elemList.append(Element(elemIDs[i], elemGeoFeatures[i], elemConnections[i]))
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'getElementsInfo() took {finishTime - startTime}s to run.')
    
    return elemList

# --------------------- Get Structural Element by Node ID -------------------- #
def nodeByID(allNodes: list[Node], 
             elemID:str, 
             timeDebug:bool=False):
    """Finding and returning the graph node in the given list of nodes, by its element ID.
    
    Parameters
    ----------
        ``allNodes (list[Node])``: A list of graph nodes.
        ``elemID (str)``: An element ID.
        ``timeDebug (bool, optional)``: Whether or not to print function timing for debug, False by default.
    
    Returns
    -------
        ``(Node | None)``: The graph node if exist in the list, otherwise None.
    """    
    startTime = timeit.default_timer()
    # Loop over all nodes (all elements)
    for node in allNodes:
        if node.element.id == elemID:
            # Return the element with the given id, found in the given list
            return node
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'nodeByID() took {finishTime - startTime}s to run.')
    
    # Return None if no element was found:
    return None

# ----------------------- Creating Edges Based on Nodes ---------------------- #
def nodesToEdges(allNodes: list[Node], 
                 timeDebug:bool=False):
    """Creating a list of graph edges based on a list of graph nodes.
    
    Parameters
    ----------
        ``allNodes (list[Node])``: A list of graph nodes.
        ``timeDebug (bool, optional)``: Whether or not to print function timing for debug, False by default.
    
    Returns
    -------
        ``(list[Edge])``: A list of graph edges.
    """    
    startTime = timeit.default_timer()
    edgesList = []
    
    # Loop over all nodes (all elements)
    for node in allNodes:
        # Loop over all elements connected to this node
        for connection in node.element.connections:
            # Creating a new edge class instance for each connection
            edge = Edge(id, node, nodeByID(allNodes, connection, timeDebug))
            if edge not in edgesList: # FIXME: Direction should be determined by structural support.
                edgesList.append(edge)
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'nodesToEdges() took {finishTime - startTime}s to run.')
    
    return edgesList

# -------------------- Visualizing a Graph using Networkx -------------------- #
def nxGraphVisualization(model: int, 
                         projectDir: str, 
                         graph, 
                         nodeDict:dict, 
                         figSave:bool=True, 
                         timeDebug:bool=False):
    """Visualizing a DGL graph using Networkx. Returns nothing.
    
    Parameters
    ----------
        ``model (int)``: Project number for the graph title.
        ``projectDir (str)``: Path of the project directory for saving the graph figure.
        ``graph (_type_)``: A DGL graph object.
        ``nodeDict (dict)``: A dictionary of labels for the graph nodes.
        ``figSave (bool, optional)``: Whether or not to save the figure, True by default.
        ``timeDebug (bool, optional)``: Whether or not to print function timing for debug, False by default.
    """
    startTime = timeit.default_timer()
    # Converting the graph to networkx graph and drawing it:
    nxGraph = graph.cpu().to_networkx()
    nx.draw_networkx(nxGraph, with_labels=True, arrowstyle='-', node_size=1000, \
                    node_color='#0091ea', edge_color='#607d8b', width=2.0, \
                    labels=nodeDict, label='Model Graph')
    # Figure settings:
    fig = plt.gcf()
    fig.suptitle(f'Project {model:03d} Graph', fontsize=20)
    fig.set_size_inches(20, 12)
    
    # Saving the graph figure if requested, otherwise just showing it:
    if figSave:
        plt.savefig(f'{projectDir}\graph.png', dpi=300)
    else:
        plt.show()
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'nxGraphVisualization() took {finishTime - startTime}s to run.')

# --------- Creating an Homogenous Graph from Nodes.csv and Edges.csv -------- #
def homoGraph(model: int, 
              projectDir: str, 
              allNodes: list[Node], 
              visualizeGraph:bool = True, 
              figSave:bool=True, 
              timeDebug:bool=False):
    """Creating an Homogenous DGL Graph based on Nodes.csv and Edges.csv files, and returns the graph.
    
    Parameters
    ----------
        ``model (int)``: Project number.
        ``projectDir (str)``: Path of the project directory to get the graph data from.
        ``allNodes (list[Node])``: A list of all nodes in the graph.
        ``visualizeGraph (bool, optional)``: Whether or not to visualize the graph, True by default.
        ``figSave (bool, optional)``: Whether or not to save the graph figure, True by default.
        ``timeDebug (bool, optional)``: Whether or not to print function timing for debug, False by default.
    
    Returns
    -------
        ``(DGLHeteroGraph)``: A DGL graph object.
    """
    startTime = timeit.default_timer()
    # Reading graph data from CSV files:
    nodesData = pd.read_csv(f'{projectDir}\\Nodes.csv')
    edgesData = pd.read_csv(f'{projectDir}\\Edges.csv')
    src = edgesData['Src ID'].to_numpy()
    dst = edgesData['Dst ID'].to_numpy()
    
    # Creating an homogenous DGL graph:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph = dgl.graph((src, dst), device = device)
    graph = dgl.add_reverse_edges(graph)
    
    # If graph visualization is enabled, make a figure:
    if visualizeGraph:
        nodeDict = {}
        for node in allNodes:
            nodeDict[node.nodeID] = node.element.id
        nxGraphVisualization(model, projectDir, graph, nodeDict, figSave, timeDebug)
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'homoGraph() took {finishTime - startTime}s to run.')
    
    return graph

# -------- Creating Nodes.csv and Edges.csv from Elements Information -------- #
def homoGraphFromElementsInfo(dataDir: str, 
                              modelCount: int, 
                              gPrint:bool = True, 
                              visualizeGraph:bool = True, 
                              figSave:bool=True, 
                              timeDebug:bool=False):
    """Creating a DGL graph for each model based on elements information given from Dynamo. Returns nothing.
    
    Parameters
    ----------
        ``dataDir (str)``: A path to a directory containing all elements information from all models.
        ``modelCount (int)``: The number of models in the dataset.
        ``gPrint (bool, optional)``: Whether or not to print each graph information, True by default.
        ``visualizeGraph (bool, optional)``: Whether or not to visualize each graph, True by default.
        ``figSave (bool, optional)``: Whether or not to save each graph figure, True by default.
        ``timeDebug (bool, optional)``: Whether or not to print function timing for debug, False by default.
    """
    startTime = timeit.default_timer()
    # List of structural element types possible:
    elementTypes = list(Element.featuresDict.keys())
    
    if gPrint:
        print('==================================================')
    
    # Loop over models data from Dynamo:
    for model in range(1, modelCount+1):
        projectDir = f'{dataDir}\\Project {model:03d}'
        
        with open(f'{projectDir}\\Nodes.csv', 'w') as csvFile1, \
            open(f'{projectDir}\\Edges.csv', 'w') as csvFile2:
            nodes = csv.writer(csvFile1)
            edges = csv.writer(csvFile2)
            # Writing header to CSV files:
            nodes.writerow(['Node ID', 'Element ID', 'Dim 1', 'Dim 2', 'Dim 3', 'Volume'])
            edges.writerow(['Src ID', 'Dst ID'])
            
            allElements = []
            # Loop over element types:
            for elementType in elementTypes:
                fileName = f'{projectDir}\\{elementType}sData.csv'
                elemList = getElementsInfo(fileName, timeDebug)
                # Getting all elements' information from all element types:
                allElements.extend(elemList)
            
            # Writing to Nodes.csv file:
            allNodes = []
            nodeIDs = list(range(len(allElements)))
            # Loop over all elements:
            for i in nodeIDs:
                # Creating a list of class Node with a node ID and an element:
                allNodes.append(Node(i, allElements[i]))
                nodes.writerow(allNodes[i].getNodeAsList())
                # FIXME: The dimension features are not in the same order for all elements.
                # TODO: The features should be normalized before learning.
            
            # Writing to Edges.csv file:
            allEdges = nodesToEdges(allNodes, timeDebug)
            # Loop over all edges:
            for i in range(len(allEdges)):
                edges.writerow(allEdges[i].getEdgeAsList())
        
        # Getting the DGL graph of each model in the dataset.
        graph = homoGraph(model, projectDir, allNodes, visualizeGraph, figSave, timeDebug)
        # TODO: Add the node features to the graph.
        
        if gPrint:
            print(f'    Graph Information of Project {model:03d}:')
            print('==================================================')
            print(f'    Number of nodes: {graph.num_nodes()}')
            print(f'    Number of edges: {graph.num_edges()}')
            print(f'    Is the graph homogenous: {graph.is_homogeneous}')
            print(f'    The graph device is: {graph.device}')
            print('==================================================')
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'homoGraphFromElementsInfo() took {finishTime - startTime}s to run.')

# ------------------------------- Main Function ------------------------------ #
def main():
    # Initial data for running the script:
    startTime = timeit.default_timer()
    workspace = os.getcwd()
    dataDir = f'{workspace}\\Dynamo'
    Element.featuresDict = {'Beam': 4, 'Column': 4, 'Slab': 5, 'Wall': 4}
    modelCount = 2
    
    # Calling the functions:
    homoGraphFromElementsInfo(dataDir, modelCount, visualizeGraph=False, timeDebug=False)
    
    # Timing the script:
    finishTime = timeit.default_timer()
    print(f'Program run in {finishTime - startTime:.3f}s')

# ------------------------------- Run as Script ------------------------------ #
if __name__ == '__main__':
    main()