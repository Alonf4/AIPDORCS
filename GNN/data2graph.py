# ---------------------------------- Header ---------------------------------- #
"""A module for converting Dynamo data from CSV files to Graphs for deep learning on structural Revit models.

Functions
---------
    ``getElementsInfo``: 
    Extracting structural elements information to a list from a given filename.
    
    ``nodeByID``: 
    Finding and returning the graph node in the given list of nodes, by its element ID.
    
    ``nodesToEdges``: 
    Creating a list of graph edges based on a list of graph nodes.
    
    ``nxGraphVisualization``: 
    Visualizing a DGL graph using Networkx. Returns nothing.
    
    ``modelAverageScore``:
    Calculating and returning the average score for a given project number, based on all answers from the Engineers' Challenge of this project.
    
    ``graphLabel``:
    Getting the label for the project based on the overall score from the Engineers' Challenge.
    
    ``homoGraph``:
    Creating an Homogenous DGL Graph based on Nodes.csv and Edges.csv files, and returns the graph.
    
    ``homoGraphFromElementsInfo``:
    Creating a DGL graph for each model based on elements information given from Dynamo. Returns nothing.
"""
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
from websiteData import *

# ------------------------- Get Elements Information ------------------------- #
def getElementsInfo(fileName:str, 
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
                    connections = list(filter(None, connections))
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
def nodeByID(allNodes:list[Node], 
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
def nodesToEdges(allNodes:list[Node], 
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
            if edge not in edgesList: # REVIEW: Direction should be determined by structural support.
                edgesList.append(edge)
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'nodesToEdges() took {finishTime - startTime}s to run.')
    
    return edgesList

# -------------------- Visualizing a Graph using Networkx -------------------- #
def nxGraphVisualization(model:int, 
                         DatabaseProjDir:str, 
                         graph, 
                         nodeDict:dict, 
                         figSave:bool=True, 
                         timeDebug:bool=False):
    """Visualizing a DGL graph using Networkx. Returns nothing.
    
    Parameters
    ----------
        ``model (int)``: Project number for the graph title.
        ``DatabaseProjDir (str)``: Path of the project directory for saving the graph figure.
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
        plt.savefig(f'{DatabaseProjDir}\graph.png', dpi=300)
    else:
        plt.show()
    fig.clf()
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'nxGraphVisualization() took {finishTime - startTime}s to run.')

# --------------------- Calculating Project Average Score -------------------- #
def modelAverageScore(model:int, 
                      feedback:int, 
                      ECFile:str, 
                      timeDebug:bool=False):
    """Calculating and returning the average score for a given project number, based on all answers from the Engineers' Challenge of this project.
    
    Parameters
    ----------
        ``model (int)``: Project number.
        ``ECFile (str)``: A path of the Engineers' Challenge CSV file.
        ``timeDebug (bool, optional)``: Whether or not to print function timing for debug, False by default.
    
    Returns
    -------
        ``(float)``: The average score for the project.
    """
    startTime = timeit.default_timer()
    # Read all data from the CSV file and sorting it:
    df = pd.read_csv(ECFile)
    df = df.sort_values(by=['Project ID'])
    df = df.reset_index(drop = True)
    
    # Converting the model number to a string for comparison:
    modelStr = f'Project {model:03d}'
    
    # Finding all scores for the given model number and feedback number:
    modelScores = []
    count = 0
    for i, project in enumerate(df.loc[:, 'Project ID'].values.tolist()):
        if project == modelStr:
            modelScores.append(df.loc[:, 'Overall Score'].values.tolist()[i])
            
            count += 1
            if count == feedback:
                feedbackScore = df.loc[:, 'Overall Score'].values.tolist()[i]
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'modelAverageScore() took {finishTime - startTime}s to run.')
    
    if feedback == 0:
        # The average score of the model:
        return sum(modelScores) / len(modelScores)
    else:
        # The specific feedback score:
        return feedbackScore

# ----------------- Graph Label by Final Layer Function Type ----------------- #
def graphLabel(model:int, 
               feedback:int, 
               ECFile:str, 
               finalLayerFunc:str, 
               threshold:float=None, 
               timeDebug:bool=False):
    """Getting the label for the project based on the overall score from the Engineers' Challenge.
    
    Parameters
    ----------
        ``model (int)``: Project number.
        ``feedback (int)``: Engineers' Challenge feedback number of a given model.
        ``ECFile (str)``: A path of the Engineers' Challenge CSV file.
        ``finalLayerFunc (str)``: A string describing the final layer function.
        ``threshold (float, optional)``: A threshold score for binary classification, by default None.
        ``timeDebug (bool, optional)``: Whether or not to print function timing for debug, False by default.
    
    Returns
    -------
        ``(float | Literal[1, 0])``: A graph label: class or score
    """
    startTime = timeit.default_timer()
    score = modelAverageScore(model, feedback, ECFile, timeDebug)
    
    match finalLayerFunc:
        case 'Regression':
            label = score
        case 'Binary Classification':
            label = 1 if score >= threshold else 0
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'graphLabel() took {finishTime - startTime}s to run.')
    
    return label

# --------- Creating an Homogenous Graph from Nodes.csv and Edges.csv -------- #
def homoGraph(model:int, 
              feedback:int, 
              DatabaseProjDir:str, 
              ECFile:str, 
              allNodes:list[Node], 
              finalLayerFunc:str, 
              threshold:float=None, 
              visualizeGraph:bool=True, 
              figSave:bool=True, 
              timeDebug:bool=False):
    """Creating an Homogenous DGL Graph based on Nodes.csv and Edges.csv files, and returns the graph.
    
    Parameters
    ----------
        ``model (int)``: Project number.
        ``feedback (int)``: Engineers' Challenge feedback number of a given model.
        ``DatabaseProjDir (str)``: A path of the project directory to get the graph data from.
        ``ECFile (str)``: A path of the Engineers' Challenge CSV file.
        ``allNodes (list[Node])``: A list of all nodes in the graph.
        ``finalLayerFunc (str)``: A string describing the final layer function.
        ``threshold (float)``: A threshold score for binary classification.
        ``visualizeGraph (bool, optional)``: Whether or not to visualize the graph, True by default.
        ``figSave (bool, optional)``: Whether or not to save the graph figure, True by default.
        ``timeDebug (bool, optional)``: Whether or not to print function timing for debug, False by default.
    
    Returns
    -------
        ``(DGLHeteroGraph)``: A DGL graph object.
        ``(float)``: A label for the graph - overall score.
    """
    startTime = timeit.default_timer()
    nodesNumber = "" if feedback == 0 else feedback
    # Reading graph data from CSV files:
    nodesData = pd.read_csv(f'{DatabaseProjDir}\\Nodes{nodesNumber}.csv')
    edgesData = pd.read_csv(f'{DatabaseProjDir}\\Edges.csv')
    src = edgesData['Src ID'].to_numpy()
    dst = edgesData['Dst ID'].to_numpy()
    
    # Getting graph features depending on the experiment (models vs Engineers' Challenge feedbacks):
    if feedback == 0:
        features = nodesData.loc[:, ['Dim 1','Dim 2','Dim 3','Volume']].to_numpy()
    else:
        websiteFeatures = list(Element.commentsDict.keys()) + ['Element Points']
        features = nodesData.loc[:, ['Dim 1','Dim 2','Dim 3','Volume'] + websiteFeatures].to_numpy()
    
    # Creating an homogenous DGL graph:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    graph = dgl.graph((src, dst), device = device)
    graph = dgl.add_reverse_edges(graph)
    features = torch.from_numpy(features).to(device)
    graph.ndata['feat'] = features # REVIEW: Check if getting node features correctly.
    label = graphLabel(model, feedback, ECFile, finalLayerFunc, threshold, timeDebug)
    
    # If graph visualization is enabled, make a figure:
    if visualizeGraph:
        nodeDict = {}
        for node in allNodes:
            nodeDict[node.nodeID] = node.element.id
        nxGraphVisualization(model, DatabaseProjDir, graph, nodeDict, figSave, timeDebug)
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'homoGraph() took {finishTime - startTime}s to run.')
    
    return graph, label

# -------- Creating Nodes.csv and Edges.csv from Elements Information -------- #
def homoGraphFromElementsInfo(websiteDir:str, 
                              dynamoDir:str, 
                              dataDir:str, 
                              ECFile:str, 
                              modelCount:int, 
                              finalLayerFunc:str, 
                              threshold:float=None, 
                              gPrint:bool = True, 
                              visualizeGraph:bool = True, 
                              figSave:bool=True, 
                              timeDebug:bool=False):
    """Creating a DGL graph for each model based on elements information given from Dynamo. Returns nothing.
    
    Parameters
    ----------
        ``websiteDir (str)`` : A path to the website directory containing the Engineers' Challenge CSV file.
        ``dynamoDir (str)``: A path to a directory containing all elements information from all models.
        ``dataDir (str)``: A path to a directory to save all graph information for each model.
        ``ECFile (str)``: A path of the Engineers' Challenge CSV file.
        ``modelCount (int)``: The number of models in the dataset.
        ``finalLayerFunc (str)``: A string describing the final layer function.
        ``threshold (float)``: A threshold score for binary classification.
        ``gPrint (bool, optional)``: Whether or not to print each graph information, True by default.
        ``visualizeGraph (bool, optional)``: Whether or not to visualize each graph, True by default.
        ``figSave (bool, optional)``: Whether or not to save each graph figure, True by default.
        ``timeDebug (bool, optional)``: Whether or not to print function timing for debug, False by default.
    """
    startTime = timeit.default_timer()
    graphList = []
    graphsList = []
    labelList = []
    labelsList = []
    # List of structural element types possible:
    elementTypes = list(Element.featuresDict.keys())
    
    if gPrint:
        print('==================================================')
    
    # Loop over models data from Dynamo:
    for model in range(1, modelCount+1):
        DynamoProjDir = f'{dynamoDir}\\Project {model:03d}'
        DatabaseProjDir = f'{dataDir}\\Project {model:03d}'
        # Create a new directory if it doesn't already exist:
        if not os.path.isdir(DatabaseProjDir):
            os.makedirs(DatabaseProjDir)
        
        with open(f'{DatabaseProjDir}\\Nodes.csv', 'w') as csvFile1, \
            open(f'{DatabaseProjDir}\\Edges.csv', 'w') as csvFile2:
            nodes = csv.writer(csvFile1)
            edges = csv.writer(csvFile2)
            # Writing header to CSV files:
            nodes.writerow(['Node ID', 'Element ID', 'Dim 1', 'Dim 2', 'Dim 3', 'Volume'])
            edges.writerow(['Src ID', 'Dst ID'])
            
            allElements = []
            # Loop over element types:
            for elementType in elementTypes:
                fileName = f'{DynamoProjDir}\\{elementType}sData.csv'
                elemList = getElementsInfo(fileName, timeDebug)
                # Getting all elements' information from all element types:
                allElements.extend(elemList)
            
            # Writing geometric features to Nodes.csv file:
            allNodes = []
            nodeIDs = list(range(len(allElements)))
            # Loop over all elements:
            for i in nodeIDs:
                # Creating a list of class Node with a node ID and an element:
                allNodes.append(Node(i, allElements[i]))
                nodes.writerow(allNodes[i].NodeGeoFeaturesAsList())
                # FIXME: The dimension features are not in the same order for all elements.
                # TODO: The features should be normalized before learning.
            
            # Writing to Edges.csv file:
            allEdges = nodesToEdges(allNodes, timeDebug)
            # Loop over all edges:
            for i in range(len(allEdges)):
                edges.writerow(allEdges[i].getEdgeAsList())
        
        # Writing all features to Nodes#.csv file:
        feedbacks = allFeaturesDataNodes(websiteDir, DatabaseProjDir, model, allElements)
        
        # Getting the DGL graph of each model in the dataset:
        graph, label = homoGraph(model, 0, DatabaseProjDir, ECFile, allNodes, finalLayerFunc, threshold, visualizeGraph, figSave, timeDebug)
        graphList.append(graph)
        labelList.append(label)
        
        # Getting the DGL graphs of each model and feedback in the dataset:
        for feedback in range(1, feedbacks+1):
            graphs, labels = homoGraph(model, feedback, DatabaseProjDir, ECFile, allNodes, finalLayerFunc, threshold, visualizeGraph, figSave, timeDebug)
            graphsList.append(graphs)
            labelsList.append(labels)
        
        if gPrint:
            print(f'    Graph Information of Project {model:03d}:')
            print('==================================================')
            print(f'    Number of nodes: {graph.num_nodes()}')
            print(f'    Number of edges: {graph.num_edges()}')
            print(f'    Graph label: {label}')
            print(f'    Is the graph homogenous: {graph.is_homogeneous}')
            print(f'    The graph device is: {graph.device}')
            print('==================================================')
    
    graphLabels = {"glabel": torch.tensor(labelList)}
    dgl.save_graphs(f'{dataDir}\\dataset1.bin', graphList, graphLabels)
    
    graphsLabels = {"glabel": torch.tensor(labelsList)}
    dgl.save_graphs(f'{dataDir}\\dataset2.bin', graphsList, graphsLabels)
    
    # Timing function debug:
    finishTime = timeit.default_timer()
    if timeDebug:
        print(f'homoGraphFromElementsInfo() took {finishTime - startTime}s to run.')

# ------------------------------- Main Function ------------------------------ #
def main():
    # Initial data for running the script:
    startTime = timeit.default_timer()
    workspace = os.getcwd()
    websiteDir = f'{workspace}\\Website'
    dynamoDir = f'{workspace}\\Dynamo'
    dataDir = f'{workspace}\\Database'
    ECFile = f'{workspace}\\Website\\EngineersChallenge.csv'
    Element.featuresDict = {'Beam': 4, 'Column': 4, 'Slab': 5, 'Wall': 4}
    
    Element.commentsDict = {"Cross-section dimensions are too small": 0, 
                            "Cross-section dimensions are too big": 1, 
                            "Element too slender (long for its cross-section)": 2, 
                            "Element with too wide a span": 3, 
                            "Element difficult to execute on-site": 4, 
                            "Element too expensive to execute": 5, 
                            "Element's position in space is not optimal": 6, 
                            "Redundant element": 7, 
                            "Other": 8}
    
    generalCommentsDict = {"Insufficient vertical supports (columns/walls)": 0, 
                           "An excessive amount of vertical supports (columns/walls)": 1, 
                           "Risk of systems clashing (e.g. beams are too deep)": 2, 
                           "Non-optimal general structural scheme": 3, 
                           "Insufficient structural cores": 4, 
                           "Non-optimal position of structural cores in space": 5, 
                           "Uneconomic alternative": 6, 
                           "Difficult to implement": 7, 
                           "Another construction method is preferable (conventional/industrialized)": 8, 
                           "Another slab alternative is preferable (waffle slab/hollow-core slab...)": 9, 
                           "Other": 10} # TODO: Add global features.
    
    experienceMultiplier = {"No Experience": 1, 
                            "0-2 Years": 2, 
                            "2-5 Years": 3, 
                            "5-10 Years": 4, 
                            "10-20 Years": 5, 
                            "20-30 Years": 6, 
                            "30+ Years": 7} # TODO: Add weight multiplier
    
    finalLayerFunc = 'Binary Classification'
    threshold = 75
    modelCount = 48
    
    # Creating an Histogram of overall scores based on the Engineers' Challenge:
    engineersChallengeHistogram(websiteDir, figSave=True)
    
    # Calling the functions:
    homoGraphFromElementsInfo(websiteDir, 
                              dynamoDir, 
                              dataDir, 
                              ECFile, 
                              modelCount, 
                              finalLayerFunc=finalLayerFunc, 
                              threshold=threshold, 
                              visualizeGraph=False, 
                              timeDebug=False)
    
    # Timing the script:
    finishTime = timeit.default_timer()
    print(f'Program run in {finishTime - startTime:.3f}s')

# ------------------------------- Run as Script ------------------------------ #
if __name__ == '__main__':
    main()

# ---------------------------------- Footer ---------------------------------- #
"""
Script Properties
-----------------
    ``Author``: Alon D. Argaman
    ``Research``: AIPDORCS - Artificially Intelligent Preliminary Design of Reinforced Concrete Structures
    ``Degree``: M.Sc. in Structural Engineering
    ``Faculty``: Civil & Environmental Engineering
    ``Institute``: Technion - Israel Institute of Technology
    
    ``Python Version``: 3.10
    ``Status``: Incomplete
    ``Runnable Script``: Yes
"""