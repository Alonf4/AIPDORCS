# ------------------------------ Modules Imports ----------------------------- #
import csv
import string

# ------------------------- Get Elements Information ------------------------- #
def getElementsInfo(fileName: str):
    """Extracting structural elements information to 3 lists from a given filename.

    Args:
        fileName (str): A path to the file.

    Returns:
        elemIDs (list): A list of element IDs.
        elemConnections (list): A list of each element's connection lists.
        elemGeoFeatures (list): A list of each element's geometric features lists.
    """    
    # Dictionary of number of features for each element type:
    featuresDict = {'Beam': 4, 'Column': 4, 'Slab': 5, 'Wall': 4}
    
    # Lists of elements information:
    elemIDs = []
    elemConnections = []
    elemGeoFeatures = []
    tempList = []
    
    # Getting the number of geometric features in the given file
    elementType = [item for item in featuresDict if item in fileName]
    featureCount = featuresDict[elementType[0]]
    
    # Getting element information from the database:
    with open(fileName, 'r') as csv_file:
        elementsFile = csv.reader(csv_file, delimiter=',')
        # Loop over lines in the file:
        for line in elementsFile:
            match line[0]:
                case 'Beam ID' | 'Column ID' | 'Slab ID' | 'Wall ID':
                    elemIDs.append(line[1])
                case 'Beam Connections' | 'Column Connections' | \
                    'Slab Connections' | 'Wall Connections':
                    elemConnections.append(line[1:])
                # Other = geometric features
                case other:
                    tempList.append(line[1])
                    # Adding all element geometric features together:
                    if len(tempList) == featureCount:
                        elemGeoFeatures.append(tempList)
                        tempList = []
    
    return elemIDs, elemConnections, elemGeoFeatures

# ------------------------------- Main Function ------------------------------ #
def main():
    elemIDs = []
    elemConnections = []
    elemGeoFeatures = []
    
    projectNumber = 2
    
    fileName = '~\\..\\Dynamo\\Project ' + f'{projectNumber:03d}' + '\\BeamsData.csv'
    elemIDs, elemConnections, elemGeoFeatures = getElementsInfo(fileName)
    
    print(elemIDs, len(elemIDs))
    print(elemConnections, len(elemConnections))
    print(elemGeoFeatures, len(elemGeoFeatures))

# ------------------------------- Run as Script ------------------------------ #
if __name__ == '__main__':
    main()