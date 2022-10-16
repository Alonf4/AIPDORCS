# ------------------------------ Modules Imports ----------------------------- #
# Official modules:
import os
import csv
import pandas as pd
import dgl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# AIPDORCS modules:
from classes import *

# ------------------------- Get Elements Information ------------------------- #
def getElementsInfo(fileName: str):
    """Extracting structural elements information to a list from a given filename.
    
    Parameters
    ----------
        ``fileName (str)``: A path to a file.
    
    Returns
    -------
        ``(list[Element])``: A list of all structural elements information.
    
    """    
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
                    elemConnections.append(list(set(line[1:])))
                
                case other: # other = geometric features
                    tempList.append(line[1])
                    # Adding all elements' geometric features together:
                    if len(tempList) == featureCount:
                        elemGeoFeatures.append(tempList)
                        tempList = []
    
    # Creating a list of all elements with all their information from the CSV files:
    for i in range(len(elemIDs)):
        elemList.append(Element(elemIDs[i], elemGeoFeatures[i], elemConnections[i]))
    
    return elemList

# --------------------- Get Structural Element by Node ID -------------------- #
def nodeByID(allNodes: list[Node], elemID:str):
    """Finding and returning the graph node in the given list of nodes, by its element ID.
    
    Parameters
    ----------
        ``allNodes (list[Node])``: A list of graph nodes.
        ``elemID (str)``: An element ID.
    
    Returns
    -------
        ``(Node | None)``: The graph node if exist in the list, otherwise None.
    """    
    # Loop over all nodes (all elements)
    for node in allNodes:
        if node.element.id == elemID:
            # Return the element with the given id, found in the given list
            return node
    
    # Return None if no element was found:
    return None

# ----------------------- Creating Edges Based on Nodes ---------------------- #
def nodesToEdges(allNodes: list[Node]):
    """Creating a list of graph edges based on a list of graph nodes.
    
    Parameters
    ----------
        ``allNodes (list[Node])``: A list of graph nodes.
    
    Returns
    -------
        ``(list[Edge])``: A list of graph edges.
    """    
    edgesList = []
    
    # Loop over all nodes (all elements)
    for node in allNodes:
        # Loop over all elements connected to this node
        for connection in node.element.connections:
            # Creating a new edge class instance for each connection
            edge = Edge(id, node, nodeByID(allNodes, connection))
            if edge not in edgesList: # FIXME: Homogenous Graph should contain edges of both directions.
                edgesList.append(edge)
    
    return edgesList

# -------------------- Visualizing a Graph using Networkx -------------------- #
def nxGraphVisualization(model: int, projectDir: str, graph, nodeDict:dict, figSave:bool = True):
    """Visualizing a DGL graph using Networkx. Returns nothing.
    
    Parameters
    ----------
        ``model (int)``: Project number for the graph title.
        ``projectDir (str)``: Path of the project directory for saving the graph figure.
        ``graph (_type_)``: A DGL graph object.
        ``nodeDict (dict)``: A dictionary of labels for the graph nodes.
        ``figSave (bool, optional)``: Whether or not to save the figure, True by default.
    """
    # Converting the graph to networkx graph and drawing it:
    nxGraph = graph.to_networkx()
    nx.draw_networkx(nxGraph, with_labels=True, arrowstyle='-', node_size=1000, \
                    node_color='#0091ea', edge_color='#607d8b', width=2.0, \
                    labels=nodeDict, label='Model Graph')
    # Figure settings:
    fig = plt.gcf()
    fig.suptitle(f'Project {model:03d} Graph', fontsize=20)
    fig.set_size_inches(20, 12)
    
    # Saving the graph figure if requested, otherwise just showing it:
    if figSave == True:
        plt.savefig(f'{projectDir}\graph.png', dpi=300)
    else:
        plt.show()

# --------- Creating an Homogenous Graph from Nodes.csv and Edges.csv -------- #
def homoGraph(model: int, projectDir: str, allNodes: list[Node], visualizeGraph:bool = True, figSave:bool = True):
    """Creating an Homogenous DGL Graph based on Nodes.csv and Edges.csv files, and returns the graph.
    
    Parameters
    ----------
        ``model (int)``: Project number.
        ``projectDir (str)``: Path of the project directory to get the graph data from.
        ``allNodes (list[Node])``: A list of all nodes in the graph.
        ``visualizeGraph (bool, optional)``: Whether or not to visualize the graph, True by default.
        ``figSave (bool, optional)``: Whether or not to save the graph figure, True by default.
    
    Returns
    -------
        ``(DGLHeteroGraph)``: A DGL graph object.
    """
    # Reading graph data from CSV files:
    nodesData = pd.read_csv(f'{projectDir}\\Nodes.csv')
    edgesData = pd.read_csv(f'{projectDir}\\Edges.csv')
    src = edgesData['Src ID'].to_numpy()
    dst = edgesData['Dst ID'].to_numpy()
    # Creating an homogenous DGL graph:
    graph = dgl.graph((src, dst))
    
    # If graph visualization is enabled, make a figure:
    if visualizeGraph == True:
        nodeDict = {}
        for node in allNodes:
            nodeDict[node.nodeID] = node.element.id
        nxGraphVisualization(model, projectDir, graph, nodeDict, figSave)
    
    return graph

# -------- Creating Nodes.csv and Edges.csv from Elements Information -------- #
def homoGraphFromElementsInfo(dataDir: str, modelCount: int, visualizeGraph:bool = True, figSave:bool = True):
    """Creating a DGL graph for each model based on elements information given from Dynamo. Returns nothing.
    
    Parameters
    ----------
        ``dataDir (str)``: A path to a directory containing all elements information from all models.
        ``modelCount (int)``: The number of models in the dataset.
        ``visualizeGraph (bool, optional)``: Whether or not to visualize each graph, True by default.
        ``figSave (bool, optional)``: Whether or not to save each graph figure, True by default.
    """
    # List of structural element types possible:
    elementTypes = list(Element.featuresDict.keys())
    
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
                elemList = getElementsInfo(fileName)
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
            allEdges = nodesToEdges(allNodes)
            # Loop over all edges:
            for i in range(len(allEdges)):
                edges.writerow(allEdges[i].getEdgeAsList())
        
        # Getting the DGL graph of each model in the dataset.
        graph = homoGraph(model, projectDir, allNodes, visualizeGraph, figSave)

# ------------------------------- Main Function ------------------------------ #
def main():
    workspace = os.getcwd()
    dataDir = f'{workspace}\\Dynamo'
    Element.featuresDict = {'Beam': 4, 'Column': 4, 'Slab': 5, 'Wall': 4}
    modelCount = 2
    
    homoGraphFromElementsInfo(dataDir, modelCount)

# ------------------------------- Run as Script ------------------------------ #
if __name__ == '__main__':
    main()