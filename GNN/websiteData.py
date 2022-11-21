import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from classes import *

def engineersChallengeHistogram(df: pd.DataFrame):
    # Getting all overall scores of the engineers' challenge:
    scores = df.loc[:, 'Overall Score'].values.tolist()
    # Getting the mean and the standard deviation of the scores:
    mean = statistics.mean(scores)
    sd = statistics.stdev(scores)
    
    # Plotting the histogram:
    plt.hist(scores, bins=10, edgecolor='white')
    # Plotting mean and standard deviation lines:
    plt.axvline(mean, color='black', linestyle='dashed', label=f'Mean: {mean:.2f}')
    plt.axvline(mean + sd, color='orange', linestyle='dashed', label=f'Standard Deviation: {sd:.2f}')
    plt.axvline(mean - sd, color='orange', linestyle='dashed')
    # Figure display settings:
    plt.title('Overall Score Histogram')
    plt.xlabel('Scores')
    plt.ylabel('Projects')
    plt.xticks(range(0, 101, 10))
    plt.legend(loc='best')
    plt.show()
    plt.clf()

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
    
    engineersChallengeHistogram(df)
    
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