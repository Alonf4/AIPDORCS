# Official modules:
import os
import csv
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
# AIPDORCS modules:
from classes import *

# ------------------------- Creating Labels Histogram ------------------------ #
def engineersChallengeHistogram(websiteDir:str, 
                                figSave:bool=True):
    """Creating an overall score histogram for all answers of the Engineers' Challenge.
    
    Parameters
    ----------
        ``websiteDir (str)``: A path to the website directory containing the Engineers' Challenge CSV file.
        ``figSave (bool, optional)``: Whether to save the figure or to only show it, True by default
    """
    fileName = f'{websiteDir}\\EngineersChallenge.csv'
    
    # Read all data from the CSV file and sorting it:
    df = pd.read_csv(fileName)
    df = df.sort_values(by=['Project ID'])
    df = df.reset_index(drop = True)
    
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

# ------------------- Evaluating Cell Values in a CSV File ------------------- #
def evalValue(value):
    """Evaluating cell values of the Engineers' Challenge and returning the right value format.
    
    Parameters
    ----------
        ``value (Any)``: A cell value, possibly in a wrong format.
    
    Returns
    -------
        ``(Any)``: A value in the right format.
    
    Examples
    --------
    >>> # Example 1: List inside a string:
    >>> cellValue = '[1, 2, 3]'
    >>> type(cellValue) == str
    >>> finalValue = evalValue(cellValue)
    >>> finalValue = [1, 2, 3]
    >>> type(finalValue) == list[int]
    >>>
    >>> # Example 2: An empty cell:
    >>> cellValue = nan
    >>> finalValue = evalValue(cellValue)
    >>> finalValue = []
    >>>
    >>> # Example 3: A number value:
    >>> cellValue = 24.12
    >>> finalValue = evalValue(cellValue)
    >>> finalValue = 24.12
    """
    # If a list is inside a string, return it as a list:
    if type(value) == str and value[0] == '[':
        return eval(value)
    # If a cell is empty and should be a list, return it as an empty list:
    if type(value) == float and np.isnan(value):
        return []
    # For every other regular cell, just return itself:
    return value

# ---------------------- Converting Comments List Format --------------------- #
def commentStrToFloat(elementComments:list[str]):
    """Converting a comment list of strings with applied comments to list of floats with binary values for all possible comments.
    
    Parameters
    ----------
        ``elementComments (list[str])``: A list of applied string comments
    
    Returns
    -------
        ``(list[float])``: A list of floats representing all comments. If applied equals to 1, otherwise 0.
    """
    comments = [0.0] * len(Element.commentsDict)
    # Replace comments with floats:
    for key, value in Element.commentsDict.items():
        if key in elementComments:
            comments[value] = 1.0
    # Return a list of floats for all comments:
    return comments

# ------------ Adding Feedback Information to Structural Elements ------------ #
def getElementsChallengeInfo(df:pd.DataFrame, 
                             row:int, 
                             allElements:list[Element]):
    """Getting elements' Engineers' Challenge information for a given feedback.
    
    Parameters
    ----------
        ``df (pd.DataFrame)``: A Pandas DataFrame containing all Engineers' Challenge information.
        ``row (int)``: A row number in the given DataFrame.
        ``allElements (list[Element])``: A list of all structural elements of the given model.
    
    Returns
    -------
        ``(list[Element])``: An updated list of structural elements with comments and scores.
    """
    # We don't want to replace the original elements:
    updatedElements = copy.deepcopy(allElements)
    # Questions 1-5 information:
    for i in range(1, 5+1):
        # If a question is not answered, skip it:
        categoryChoice = evalValue(df[f'Category Choice {i}'][row])
        if categoryChoice == 'None':
            continue
        
        # Getting a raw answer for each question:
        elementIDs = evalValue(df[f'Element ID {i}'][row])
        elementComments = evalValue(df[f'Element Comments {i}'][row])
        elementPoints = evalValue(df[f'Element Points {i}'][row])
        
        # Updating all specified elements with comments and points:
        for element in elementIDs:
            if element in updatedElements:
                i = updatedElements.index(element)
                updatedElements[i].comments = commentStrToFloat(elementComments)
                updatedElements[i].points = elementPoints
    
    # Returning all updated elements
    return updatedElements

# ----------------- Creating Nodes.csv Files for Experiment 2 ---------------- #
def allFeaturesDataNodes(websiteDir:str, 
                         DatabaseProjDir:str, 
                         model:int, 
                         allElements:list[Element]):
    """Creating Nodes#.csv files for each feedback of the Engineers' Challenge. Returns the number of files created.
    
    Parameters
    ----------
        ``websiteDir (str)``: A path to the website directory containing the Engineers' Challenge CSV file.
        ``DatabaseProjDir (str)``: A path of the database directory to write the Nodes.csv files to.
        ``model (int)``: Project number.
        ``allElements (list[Element])``: A list of all structural elements of the given model.
    Returns
    -------
        ``(int)``: The number of Nodes#.csv files created.
    """
    modelID = f'Project {model:03d}'
    fileName = f'{websiteDir}\\EngineersChallenge.csv'
    
    # Read all data from the CSV file and sorting it:
    df = pd.read_csv(fileName)
    df = df.sort_values(by=['Project ID'])
    df = df.reset_index(drop = True)
    
    answerCount = 1
    # Loop over rows in the CSV file:
    for row in df.index:
        # Getting only the given model's information:
        projectID = evalValue(df[f'Project ID'][row])
        if modelID == projectID:
            # Writing to a new Nodes.csv file for each answer for the given model:
            with open(f'{DatabaseProjDir}\\Nodes{answerCount}.csv', 'w') as csvFile:
                file = csv.writer(csvFile)
                
                # Writing header to CSV file:
                geoFeatures = ['Node ID', 'Element ID', 'Dim 1', 'Dim 2', 'Dim 3', 'Volume']
                websiteFeatures = list(Element.commentsDict.keys()) + ['Element Points']
                file.writerow(geoFeatures + websiteFeatures)
                
                # Updating the elements' Engineers' Challenge features:
                updatedElements = getElementsChallengeInfo(df, row, allElements)
                
                allNodes = []
                nodeIDs = list(range(len(updatedElements)))
                # Loop over all elements:
                for i in nodeIDs:
                    # Creating a list of class Node with a node ID and an element:
                    allNodes.append(Node(i, updatedElements[i]))
                    file.writerow(allNodes[i].NodeFullFeaturesAsList())
                    # FIXME: The dimension features are not in the same order for all elements.
                    # TODO: The features should be normalized before learning.
            
            answerCount += 1
    
    return answerCount - 1