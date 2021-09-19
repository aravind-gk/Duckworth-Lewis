import numpy as np
import pandas as pd
import math 
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import pickle

'''Function to take data from CSV file and pre-process it in proper way'''
def prepareDataFromCSV(fileName: str):
    # Import data from csv file as dataframe
    data = pd.read_csv(fileName)
    
    # Remove matches which either contain error or adversely affected by rain
    
    # Code can be uncommented when there is a pickled list of erroneus matches which should be removed from the dataa
#     with open('Removed.pkl', 'rb') as f:
#         matches_to_remove = pickle.load(f)
#     matches_to_remove = list(set(matches_to_remove))
#     data = data[~(data.Match.isin(matches_to_remove))]
    
    ## Make the data simpler by removing unnecessary columns and rows, keeping only what is required
    
    # Keep only the first innings data
    data_first = data[data.Innings == 1].copy()
    
    # Filter non-useful columns
    df1 = data_first[['Match', 'Over', 'Total.Runs', 'Innings.Total.Runs', 'Runs.Remaining', 'Wickets.in.Hand']].copy()
    
    # Create a new column called "Overs.Remaining", to be later used as the "u" parameter
    df1['Total.Overs'] = 50
    df1['Overs.Remaining'] = df1['Total.Overs'] - df1['Over']
    df1 = df1[['Match', 'Overs.Remaining', 'Runs.Remaining', 'Wickets.in.Hand']]

    # Create the rows corresponding to u = 50 and w = 10 (before the first over is played) (since it is not already included in data)
    df2 = data_first[['Match', 'Over', 'Total.Runs', 'Innings.Total.Runs', 'Runs.Remaining', 'Wickets.in.Hand']].copy()
    df2['Total.Overs'] = 50
    df2 = df2.groupby('Match').first().reset_index()
    df2['Wickets.in.Hand'] = 10
    df2 = df2[['Match', 'Total.Overs', 'Innings.Total.Runs', 'Wickets.in.Hand']]
    df2.columns = ['Match', 'Overs.Remaining', 'Runs.Remaining', 'Wickets.in.Hand']
    
    # Append the created first-over rows to our dataframe containing other rows
    df = pd.concat([df2, df1], axis = 0)
    df.sort_values(['Match', 'Overs.Remaining'], ascending = [True, False], inplace = True)
    df = df[['Overs.Remaining', 'Wickets.in.Hand', 'Runs.Remaining']]
    
    # Rename the columns for simplicity, consistent with the notations used in class
    df.columns = ['u', 'w', 'y']

    # Return both the raw dataframe of first innings and the pre-processed dataframe
    return data_first, df


'''Function to compute Z0(w), when given the first-innings data as input'''
def computeZ0(w: int, data_first: pd.core.frame.DataFrame):
    if w == 10:
        return data_first.groupby('Match').first()['Innings.Total.Runs'].mean()
    else:
        return data_first[data_first['Wickets.in.Hand'] <= w].groupby('Match').first()['Runs.Remaining'].mean()
    
def DuckworthLewis20Params(fileName: str):    
    firstInningsData, processedData = prepareDataFromCSV(fileName)
    
    # Compute Z0(w) for w = 1, ..., 10
    Z = dict()
    for w in range(1, 11):
        Z[w] = computeZ0(w, firstInningsData)
    
    # Define the function to be optimized, as "fn"
    def fn(b):
        losses = X.apply(lambda row: (row['y'] - Z[row['w']] * (1 - np.exp(-b * row['u']))) ** 2, axis = 1)
        loss = losses.sum() / len(X)
        return loss
    
    # Use scipy.optimize library's minimizer to fit the curve of fn
    B = dict()
    for w in range(10, 0, -1):
        # Filter out those datapoints with w wickets remaining, and store it in a variable called X
        X = processedData[processedData.w == w].copy()
        result = minimize(fn, 0, method = 'L-BFGS-B')
        B[w] = result.x[0]
        # print('b[{}] = {}'.format(w, B[w]))
        
    X = processedData[processedData.w > 0].copy()
    losses = X.apply(lambda row: (row['y'] - Z[row['w']] * (1 - np.exp(-B[row['w']] * row['u']))) ** 2, axis = 1)
    loss = losses.sum() / len(X)
    print('MSE Question-1: ', loss)
    return Z, B
    
def DuckworthLewis11Params(fileName: str):
    firstInningsData, processedData = prepareDataFromCSV(fileName)
    
    # Compute Z0(w) for w = 1, ..., 10
    Z = dict()
    for w in range(1, 11):
        Z[w] = computeZ0(w, firstInningsData)
        
    # define the objective function fn, to be minimized
    def fn(L):
        temp = X.copy()
        losses = temp.apply(lambda row: (row['y'] - Z[row['w']] * (1 - np.exp(-L * row['u'] / Z[row['w']]))) ** 2, axis = 1)
        loss = losses.sum() / len(temp)
        return loss
    
    # Use scipy.optimize library's minimizer to fit the curve of fn
    X = processedData[processedData.w > 0].copy() # This filter is applied, because Z[w] is almost zero, and can create errors
    result = minimize(fn, 0.35, method = 'L-BFGS-B')
    L = result.x[0]
    # print('L = ', L)
    
    losses = X.apply(lambda row: (row['y'] - Z[row['w']] * (1 - np.exp(-L * row['u'] / Z[row['w']]))) ** 2, axis = 1)
    loss = losses.sum() / len(X)
    print('MSE Question-2: ', loss)
    
    return Z, L

'''Functions for plots'''
def plotQ1(Z, B):
    # Plot Question-1 curves 
    plt.figure(figsize = (13,9))
    for w in range(10, 0, -1):
        x = np.linspace(0,51,100)
        fx = Z[w] * (1 - np.exp(-B[w] * x))
        plt.plot(x, fx, label = 'w = {}'.format(w))

    x = np.linspace(0, 51, 100)
    slope = Z[10] * (1 - np.exp(-B[10] * 50)) / 51
    fx = slope * x
    plt.plot(x, fx, label = 'straight line')    

    plt.legend(loc="upper left")
    plt.suptitle('Question-1: Run production function', fontsize = 'xx-large')
    plt.xlabel('Overs Remaining', fontsize = 'x-large')
    plt.ylabel('Average runs obtainable', fontsize = 'x-large')
    plt.show()
    return
    
def plotQ2(Z, L):
    # Plot Question-2 curves 
    plt.figure(figsize = (13,9))
    for w in range(10, 0, -1):
        x = np.linspace(0,51,100)
        fx = Z[w] * (1 - np.exp(-L * x / Z[w]))
        plt.plot(x, fx, label = 'w = {}'.format(w))

    # plotting a straight line for reference
    x = np.linspace(0, 51, 100)
    slope = Z[10] * (1 - np.exp(-L * 50 / Z[10])) / 51
    fx = slope * x
    plt.plot(x, fx, label = 'straight line')

    plt.legend(loc="upper left")
    plt.suptitle('Question-2: Run production function', fontsize = 'xx-large')
    plt.xlabel('Overs Remaining', fontsize = 'x-large')
    plt.ylabel('Average runs obtainable', fontsize = 'x-large')
    plt.show()
    return
    
def computeSlopeQ1(Z, B):
    for w in range(1, 11):
        # for w wickets, compute slope at u = 0
        slope = Z[w] * B[w]
        print('slope at w = {}: {}'.format(w, round(slope, 5)))
    return

def computeSlopeQ2(Z, L):
    for w in range(1, 11):
        # for w wickets, compute slope at u = 0
        slope = L
        print('slope at w = {}: {}'.format(w, round(slope, 5)))
    return