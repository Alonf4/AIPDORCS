import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from classes import *

def engineersChallengeHistogram(df: pd.DataFrame, 
                                websiteDir: str, 
                                figSave:bool=True):
    # Getting all overall scores of the engineers' challenge:
    scores = df.loc[:, 'Overall Score'].values.tolist()
    # Getting the mean, the median and the standard deviation of the scores:
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    sd = statistics.stdev(scores)
    
    # Plotting the histogram:
    _, _, bars = plt.hist(scores, bins=10, edgecolor='white')
    # Plotting mean and standard deviation lines:
    plt.axvline(mean, color='black', linestyle='dashed', label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='orange', linestyle='dashed', label=f'Median: {median:.2f}')
    plt.axvline(mean + sd, color='gray', linestyle='dashed', label=f'Standard Deviation: {sd:.2f}')
    plt.axvline(mean - sd, color='gray', linestyle='dashed')
    # Figure display settings:
    plt.title('Overall Score Histogram')
    plt.bar_label(bars)
    plt.xlabel('Scores')
    plt.ylabel('Projects')
    plt.xticks(range(0, 101, 10))
    plt.legend(loc='best')
    
    # Saving the histogram figure if requested, otherwise just showing it:
    if figSave:
        plt.savefig(f'{websiteDir}\histogram.png', dpi=300)
    else:
        plt.show()
    
    # Clearing the figure:
    plt.clf()

def getElementsChallengeInfo(websiteDir: str, 
                             figSave:bool=True):
    fileName = f'{websiteDir}\\EngineersChallenge.csv'
    # Lists of elements information:
    projectIDs = []
    elemIDs = []
    elemComments = []
    elemScores = []
    
    # Read all data from the CSV file and sorting it:
    df = pd.read_csv(fileName)
    df = df.sort_values(by=['Project ID'])
    df = df.reset_index(drop = True)
    
    engineersChallengeHistogram(df, websiteDir, figSave)
    
    for model in df.loc[:, 'Project ID'].values.tolist():
        print(model)
        

def main():
    # Initial data for running the script:
    workspace = os.getcwd()
    websiteDir = f'{workspace}\\Website'
    Element.featuresDict = {'Beam': 4, 'Column': 4, 'Slab': 5, 'Wall': 4}
    modelCount = 1
    
    getElementsChallengeInfo(websiteDir, figSave=True)

if __name__ == '__main__':
    main()