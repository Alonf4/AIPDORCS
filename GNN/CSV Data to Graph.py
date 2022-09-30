# ------------------------------ Modules Imports ----------------------------- #
import csv

# ------------------------- Get Elements Information ------------------------- #
def getElementsInfo(fileName):
    featuresDict = {'Beam': 4, 'Column': 4, 'Slab': 5, 'Wall': 4}
    
    # Lists of elements information:
    elementIDs = []
    elementConnections = []
    elementGeometricFeatures = []
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
                    elementIDs.append(line[1])
                case 'Beam Connections' | 'Column Connections' | \
                    'Slab Connections' | 'Wall Connections':
                    elementConnections.append(line[1:])
                # Other = geometric features
                case other:
                    tempList.append(line[1])
                    # Adding all element geometric features together:
                    if len(tempList) == featureCount:
                        elementGeometricFeatures.append(tempList)
                        tempList = []
    
    return elementIDs, elementConnections, elementGeometricFeatures

# ------------------------------- Main Function ------------------------------ #
def main():
    elements = []
    elementsConnections = []
    elementsGeometricFeatures = []
    
    fileName = '~\\..\\Dynamo\\Project 001\\ColumnsData.csv'
    elements, elementsConnections, elementsGeometricFeatures = getElementsInfo(fileName)
    
    print(elements, len(elements))
    print(elementsConnections, len(elementsConnections))
    print(elementsGeometricFeatures, len(elementsGeometricFeatures))

# ------------------------------- Run as Script ------------------------------ #
if __name__ == '__main__':
    main()