import pandas as pd
import numpy as np
import random
import altair as alt
import plotly.graph_objects as go
import requests
import inspect
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import plotly.express as px
import pyvis
from pyvis import network as net
from itertools import combinations
from copy import deepcopy

# specify a list of columns to be binned and labeled
binned_cols = ['danceability',
 'energy',
 'loudness',
 'speechiness',
 'acousticness',
 'instrumentalness',
 'liveness',
 'valence',
 'tempo',
 'duration_ms']


def binner(dataframe, columns):
    '''takes in a datafram and columns you want to bin into 3
    outputs a new datafram with the binned columns added'''
    # now make a copy of the dataframe
    binned_data = dataframe.copy()
    # make three different bins and label them
    labels = ['l', 'm', 'h']
    bin_count = len(labels)
    # loop over the columns and 'cut' the data, returning new columns with the individual rows labeled
    # note that here we need to `drop` items that might be duplicated across bins
    for column in columns:
        binned_data[f"{column}_q_binned"] = pd.qcut(dataframe[column],
                                                     q=bin_count,
                                                    labels = labels,
                                                     duplicates='drop')
    return binned_data

def add_name_column(dataframe):
    '''adds column with playlist owner's name. Note: this only works if the input dataframe
    has a column 'playlist_title' where the first part of the title is a person's name followed
    by an underscore'''

    # gets everything before the '_' in the playlist_title column
    dataframe["person"] = dataframe["playlist_title"].str.split('_', expand=True)[0]
    return dataframe


#makes rule for binning preferences, threshold changeable with a default
# Example: If someone has below 8 songs with low valence, they have a low preference for low valence
def binRule(num, high_low = [8, 12]):
    if num < high_low[0]: # return low if count is below bottom threshold
        return "l"
    elif num > high_low[1]: # return high if count is above top threshold
        return "h"
    else:
        return "m"

# and finally, as a function
def create_feature_bin_counts_df(original_data, features):
    '''creates a dataframe with rows being people and columns being three letter frequency signatures (like 'lmm')
    for each music feature that has been binned.

    Note: original_data must have column "person" and columns "{feature_name}_q_binned"'''

    preferences = pd.DataFrame()

    # adds name column
    names = list(set(original_data['person'].to_list()))
    preferences['name'] = names


    # adds columns corresponding to features where the values are the person's preference signature for that feature
    for feature in features: #loops through music features

        # creates a list of the count of songs that were in the 'low' bin for the given music feature
        lows = [original_data[original_data['person'] == name]
                [f'{feature}_q_binned'].value_counts().to_dict()['l'] for name in names]

        # creates a similar list for count of songs in the 'medium' bin
        mediums = [original_data[original_data['person'] == name]
                [f'{feature}_q_binned'].value_counts().to_dict()['m'] for name in names]

        # creates a similar list for count of songs in the 'high' bin
        highs = [original_data[original_data['person'] == name]
                [f'{feature}_q_binned'].value_counts().to_dict()['h'] for name in names]

        # creates an empty list for the preference signatures
        list_of_signatures = []

        # creates 'preference signatures' like 'lmm' based on the frequence of each music feature
        # bin in the person's playlist
        for i in range(len(lows)):
            list_of_signatures.append(binRule(lows[i])+ binRule(mediums[i])+ binRule(highs[i]))

        # adds a column to the dataframe with the preference signatures for that feature
        preferences[feature] = list_of_signatures

    return preferences





our_data = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vRuxCwGLYB351Dzh7yusurYNh7lMtF-VdVqnAAlaO6jgmg1dpCR5LheVjjQFIlbjwA5I3Toi2s1u1nL/pub?output=csv')
#pd.read_csv('https://bit.ly/our_spotify_list_audio_data')

our_data_binned = binner(our_data, binned_cols)

our_data_binned = add_name_column(our_data_binned)

# the musical features that we binned
features = ['danceability',
 'energy',
 'loudness',
 'speechiness',
 'acousticness',
 'instrumentalness',
 'liveness',
 'valence',
 'tempo',
 'duration_ms']

preferences = create_feature_bin_counts_df(our_data_binned, features)
#preferences.head(30)

# Creating an HTML node
def create_node_html(node: str, source_df: pd.DataFrame, node_col: str):
    rows = source_df.loc[source_df[node_col] == node].itertuples()
    html_lis = []
    for r in rows:
        html_lis.append(f"""<li>Artist: {r.artist}<br>
                                Playlist: {r.playlist_name}<br>"""
                       )
    html_ul = f"""<ul>{''.join(html_lis)}</ul>"""
    return html_ul

# Adding nodes from an Edgelist
def add_nodes_from_edgelist(edge_list: list,
                               source_df: pd.DataFrame,
                               graph: nx.Graph,
                               node_col: str):
    graph = deepcopy(graph)
    node_list = pd.Series(edge_list).apply(pd.Series).stack().unique()
    for n in node_list:
        graph.add_node(n, title=create_node_html(n, source_df, node_col), spring_length=1000)
    return graph

# Adding Louvain Communities
def add_communities(G):
    G = deepcopy(G)
    partition = community_louvain.best_partition(G)
    nx.set_node_attributes(G, partition, "group")
    return G

def choose_network(df, grouping_col, chosen_word, output_file_name, output_width=800):

    # creating unique pairs
    output_grouped = df.groupby([grouping_col])[chosen_word].apply(list).reset_index()
    pairs = output_grouped[chosen_word].apply(lambda x: list(combinations(x, 2)))

    pairs2 = pairs.explode().dropna()
    print(pairs2)
    unique_pairs = pairs.explode().dropna().unique()

    # creating a new Graph
    pyvis_graph = net.Network(notebook=True, width=output_width, height="1000", bgcolor="black", font_color="white")
    G = nx.Graph()
    # adding nodes
    try:
        G = add_nodes_from_edgelist(edge_list=unique_pairs, source_df=df, graph=G, node_col=chosen_word)
    except Exception as e:
        print(e)
    # add edges
    G.add_edges_from(unique_pairs)
    # find communities
    G = add_communities(G)
    pyvis_graph.from_nx(G)
    pyvis_graph.show(output_file_name)

    return pairs



louvain_network = choose_network(preferences, 'energy', 'name', 'preferences.html')

# Makes a dataframe showing number of shared signatures between each person
synergy = pd.DataFrame()
# Get list of student names from our prior dataframe
names = list(set(our_data_binned['person'].to_list()))
# Add a new column in 'synergy' dataframe for the names
synergy['name'] = names


# for each row/person in the preferences dataframe, compare that row to all other rows/people
# in the dataframe, using 'sum' to count the number of signature matches.  Store each comparison
# sum as an entry in the 'counts' list, and then use this counts list to create a new column
# in the synergy dataframe.
for r in range(len(preferences)):
    counts = []
    for c in range(len(preferences)):
        counts.append(sum(preferences.loc[r,:] == preferences.loc[c,:]))
    synergy[r] = counts


# rename column names to be names instead of numbers
for i in range(len(synergy)):
    synergy.rename(columns={i: str(names[i])}, inplace=True)
# rename row indices to be names instead of numbers
for i in range(len(synergy)):
    synergy.rename(index={i: str(names[i])}, inplace=True)

# Remove the first column of 'names' (since names are now the indices)
synergy.drop(columns = "name", inplace=True)
#synergy

#Uses https://python-graph-gallery.com/327-network-from-correlation-matrix/
# Creates a network of things that have more than 'x' same signatures

# Set threshold
x = 3

# Transpose similarity matrix (Synergy dataframe) into a 'long' verison listing
# each pair of students and their similarity
links = synergy.stack().reset_index()
links.columns = ['Person1', 'Person2', 'value']

# remove any pairs that are the person linked to themselves and any pairs that do not meet
# the threshold x
links_filtered=links.loc[ (links['value'] > x) & (links['Person1'] != links['Person2']) ]

# create the network
G=nx.from_pandas_edgelist(links_filtered, 'Person1', 'Person2')

# draw the network
#nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)






# Uses https://stackoverflow.com/questions/62935983/vary-thickness-of-edges-based-on-weight-in-networkx
# This creates a weighted network, where the thickness of the edges corresponds to the number of same signatures shared by people
# There are a bunch of options for setting 'pos' -- see options on left of https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.kamada_kawai_layout.html

# Set threshold
x = 2


# Filter pairs to remove pairs below threshold and same-person pairs (same as creation
# of link_filtered above, but called links_clean to differentiate the two network creation
# processes)
links_clean=links.loc[ (links['value'] > x) & (links['Person1'] != links['Person2']) ]
links_clean

# create a list of tuples (node 1, node 2, weight) for all connections
weighted_edges = []
for r in range(len(links_clean)):
    weighted_edges.append((links_clean.iloc[r,0], links_clean.iloc[r,1], links_clean.iloc[r,2]))

# make graph and add weighted edges to the graph based on the tuples created above
graph = nx.Graph()
graph.add_weighted_edges_from(weighted_edges)

# set widths based on the edge attributes encoded by the weighted edges added above
# and set positions (uses kamada_kawai layout, but lots of options in link linked above)
# and set nodelist from the nodes added to the graph created above
widths = nx.get_edge_attributes(graph, "weight")
pos = nx.kamada_kawai_layout(graph)
nodelist = graph.nodes()

# Actually draw the network by drawing the nodes, then the edge, and then the labels
'''
nx.draw_networkx_nodes(graph,pos,
                       nodelist=nodelist,
                       node_size=400,
                       node_color='black',
                       alpha=0.7)
nx.draw_networkx_edges(graph,pos,
                       edgelist = widths.keys(),
                       width=list(widths.values()),
                       edge_color='darkblue',
                       alpha=0.3)
nx.draw_networkx_labels(graph, pos=pos,
                        labels=dict(zip(nodelist,nodelist)),
                        font_size=8,
                        font_color='white')

# set it to not show a box around the network, and the show the drawn network
plt.box(False)
plt.show()
'''

