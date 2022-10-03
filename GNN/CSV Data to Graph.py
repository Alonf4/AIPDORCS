# ------------------------------ Modules Imports ----------------------------- #
import os
import csv
import pandas as pd
import dgl
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Get Elements Information ------------------------- #
def getElementsInfo(fileName: str):
    """Extracting structural elements information to 3 lists from a given filename.

    Parameters
    ----------
        ``fileName (str)``: A path to the file.

    Returns
    ----------
        ``elemIDs (list)``: A list of element IDs.
        ``elemConnections (list)``: A list of each element's connection lists.
        ``elemGeoFeatures (list)``: A list of each element's geometric features lists.
    """    
    # Dictionary of number of features for each element type:
    featuresDict = {'Beam': 4, 'Column': 4, 'Slab': 5, 'Wall': 4}
    
    # Lists of elements information:
    elemIDs = []
    elemConnections = []
    elemGeoFeatures = []
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
                    elemConnections.append(line[1:])
                
                case other: # other = geometric features
                    tempList.append(line[1])
                    # Adding all elements' geometric features together:
                    if len(tempList) == featureCount:
                        elemGeoFeatures.append(tempList)
                        tempList = []
    
    return elemIDs, elemConnections, elemGeoFeatures

# ------------------ Build a graph from Elements Information ----------------- #
def homoGraphFromElementsInfo(dataDir: str, modelsCount: int):
    # List of structural element types possible:
    elementTypes = ['Beam', 'Column', 'Slab', 'Wall']
    
    # Loop over models data from Dynamo:
    for model in range(1, modelsCount+1):
        projectDir = f'{dataDir}\\Project {model:03d}'
        
        with open(f'{projectDir}\\Nodes.csv', 'w') as csvFile1, \
             open(f'{projectDir}\\Edges.csv', 'w') as csvFile2:
            nodes = csv.writer(csvFile1)
            edges = csv.writer(csvFile2)
            nodes.writerow(['Node ID'])
            edges.writerow(['Edge ID', 'Src ID', 'Dst ID'])
            
            # Loop over element types:
            for elementType in elementTypes:
                fileName = f'{projectDir}\\{elementType}sData.csv'
                elemIDs, elemConnections, elemGeoFeatures = getElementsInfo(fileName)
                for i in range(0, len(elemIDs)):
                    nodes.writerow([elemIDs[i]])
                    
                    # TODO: Write to edges file
                    # TODO: Add features to files
        
        nodesData = pd.read_csv(f'{projectDir}\\Nodes.csv')
        edgesData = pd.read_csv(f'{projectDir}\\Edges.csv')
        # src = edgesData['Src ID'].to_numpy()
        # dst = edgesData['Dst ID'].to_numpy()
        
        # g = dgl.graph((src, dst))
        
        # nx_g = g.to_networkx()
        # nx.draw(nx_g, with_labels=True)
        # plt.show()

# ------------------------------- Main Function ------------------------------ #
def main():
    # Inputting directory path and number of models:
    # print('Please type a directory path containing all projects data from Dynamo', 
    #       'or leave empty to use the default path "~\\..\\Dynamo".')
    # dataDir = input('Type a directory path here: ') or '~\\..\\Dynamo'
    # print('Please enter the number of models you have in this directory.')
    # modelsCount = int(input('Type an integer number of models: '))
    workspace = os.getcwd()
    dataDir = f'{workspace}\\Dynamo'
    modelsCount = 2
    
    homoGraphFromElementsInfo(dataDir, modelsCount)

# ------------------------------- Run as Script ------------------------------ #
if __name__ == '__main__':
    main()