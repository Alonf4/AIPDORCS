import os
import csv
import pandas as pd
from classes import *

def getElementsChallengeInfo(fileName: str, 
                    timeDebug:bool=False):
    # Lists of elements information:
    projectIDs = []
    elemIDs = []
    elemComments = []
    elemScores = []
    
    # Read all data from the CSV file and sorting it:
    df = pd.read_csv(fileName)
    df = df.sort_values(by=['Project ID'])
    df = df.reset_index(drop = True)
    
    for model in df.loc[:, 'Project ID'].values.tolist():
        print(model)
        

def main():
    # Initial data for running the script:
    workspace = os.getcwd()
    websiteDir = f'{workspace}\\Website'
    fileName = f'{websiteDir}\\EngineersChallenge.csv'
    Element.featuresDict = {'Beam': 4, 'Column': 4, 'Slab': 5, 'Wall': 4}
    modelCount = 1
    getElementsChallengeInfo(fileName)
    

if __name__ == '__main__':
    main()