import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import networkx as nx
import warnings

def download_stock_dfs(sample_stocks):
    """Downloads stock dataframes containing OHLCV data for list of tickers using yfinance. 
    """
    data = yf.download(  
            tickers = sample_stocks,

            # use "period" instead of start/end
            # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            period = "5y",

            # fetch data by interval (including intraday if period < 60 days)
            # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            # (optional, default is '1d')
            interval = "1d",

            group_by = 'ticker'
        )
    return data

def show_graph_with_labels(adjacency_matrix, mylabels, graph_type='network'):
    """Plots Minimum Spanning Tree graph with labels
    mylabels is a list of strings containing ticker names. 
    """
    rows, cols = np.where(adjacency_matrix > 0)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    labeldict = {}
    for i in range(len(mylabels)):
        labeldict[i] = mylabels[i]
    if len(mylabels) > 30:
        plt.figure(3,figsize=(12,12)) 
    elif len(mylabels) > 20:
        plt.figure(3,figsize=(8,8)) 
    if graph_type == 'circle':
        nx.draw_circular(gr, node_size=550, labels=labeldict, with_labels=True, node_color='#D1D0CE')
    else:
        nx.draw(gr, node_size=550, labels=labeldict, with_labels=True, node_color='#D1D0CE')
    plt.show()

def shortest_edge(matrix):
    """Returns smallest cell value in matrix - also the shortest edge"""
    result = np.where(matrix == np.amin(matrix))
    ti = result[0][0]
    tj = result[1][0]
    return (ti, tj)

def shortest_edges(matrix):
    """Returns an ordered list of edges or cell values, starting with the smallest. 
    The param matrix is modified to keep track of added edges. 
    """
    matrix_copy = matrix
    visited_positions = []
    n = len(matrix) # n is number of nodes or stocks 
    while len(visited_positions) < ((n* n-1)/2)-1:
        edge = shortest_edge(matrix_copy)
        visited_positions.append(edge)
        matrix_copy[edge[0]][edge[1]] = np.inf
        matrix_copy[edge[1]][edge[0]] = np.inf
    return visited_positions

def build_mst(adjancey_matrix):
    """Returns the Minimum Spanning Tree given an adjacency matrix of distances by applying Kruskall's algorithm. 
 
    Cell (i, j) > 0  and equals distance if connected, and NaN otherwise.
    """
    matrix_copy = np.copy(adjancey_matrix) 
    ordered_edges = shortest_edges(matrix_copy)
    matrix_len = len(matrix_copy)
    new_matrix = np.zeros((matrix_len, matrix_len))
    set_groups = []
    for n in range(len(ordered_edges)):
        i = ordered_edges[n][0]
        j = ordered_edges[n][1]
        if not check_connected((i, j), new_matrix, set_groups):
            new_matrix[i][j] = matrix_copy[i][j]
            new_matrix[j][i] = matrix_copy[i][j]
            set_groups = add_edge_to_set((i,j), set_groups)
    new_matrix = (new_matrix < np.inf)

    final_matrix = np.ma.masked_where(new_matrix, adjancey_matrix)
    # final_matrix = final_matrix.filled(np.nan)
    final_matrix = final_matrix.filled(0)
    return final_matrix           

def add_edge_to_set(edge, set_groups):
    """Adds edge (i, j) as a connection, by adding to to a set group of connected edges
    set_groups acts as a Disjoin Set Union.

    (1) If only node i is found, add j to the same set and vice versa. 
    (2) If no sets exist, create new set with i and j in it. 
    (3) If node i and j exist in different sets, merge the sets together. 
    """
    if not set_groups: # (2)
        new_set = {edge[0], edge[1]}
        set_groups.append(new_set)
        return set_groups
    for i in range(len(set_groups)):
        if edge[0] in set_groups[i]:
            for k in range(len(set_groups)):
                if k !=i and edge[1] in set_groups[k]: # (3) node i and j in different sets
                    new_set = set_groups[k].union(set_groups[i])
                    del set_groups[i]
                    del set_groups[k-1]
                    set_groups.append(new_set)
                    return set_groups
            set_groups[i].add(edge[1]) # (1) node i found but node j not in any sets yet
            return set_groups
        if edge[1] in set_groups[i]:
            for k in range(len(set_groups)): 
                if k !=i and edge[0] in set_groups[k]: # (3) merging two sets
                    new_set = set_groups[k].union(set_groups[i])
                    del set_groups[i]
                    del set_groups[k-1]
                    set_groups.append(new_set)
                    return set_groups
            set_groups[i].add(edge[0]) # (1) 
            return set_groups
        
    new_set = {edge[0], edge[1]} # (2) new set must be created
    set_groups.append(new_set)
    return set_groups

def check_connected(edge, new_matrix, set_groups):
    """Checks if edge (i, j) is already connected to existing connected groups of nodes.
    """
    if not set_groups: # no set groups exist, so edge is not connected
        return False
    if new_matrix[edge[0]][edge[1]] > 0: # edge is already connected directly
        return True
    for group in set_groups: # node i and j are connected directly or via other nodes
        if edge[0] in group and edge[1] in group:
            return True
    else: 
        return False

def prep_dfs_for_mst(dfs):
    """Takes in list of dataframes of OHLCV data and returns a matrix of distances.

    (1) Calculates the log return for each dataframe
    (2) Calculates the correlation matrix for all log returns
    (3) Calculates the distance matrix

    Edges from a node to itself (diagonal) are np.inf for later comparisons. 
    """
    log_returns=[]
    for df in dfs:
        df['logReturns'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
        log_returns.append(np.delete(df['logReturns'].to_numpy(), 0))
    correlation = np.corrcoef(log_returns)
    distance = np.sqrt(2.0 * (1.0- correlation)) 
    np.fill_diagonal(distance, np.inf)
    return distance

def plot_mst_from_dfs(stocks):
    """Returns the minimum spanning generated from a list of stocks for past 5 years """
    dfs = download_stock_dfs(stocks)
    log_returns= []
    for stock in stocks:
        dfs[stock, 'logReturn'] = np.log(dfs[stock, 'Close']) - np.log(dfs[stock, 'Close'].shift(1))
        series = np.delete(dfs[stock, 'logReturn'].to_numpy(), 0)
        log_returns.append(series)
        if np.isnan(series).any() == True:
            warnings.warn("One of the dataframes is a different length causing NaN's to appear in {} dataframe".format(stock))
    correlation = np.corrcoef(log_returns)
    distance = np.sqrt(2.0 * (1.0- correlation)) 
    np.fill_diagonal(distance, np.inf)
    return build_mst(distance)

