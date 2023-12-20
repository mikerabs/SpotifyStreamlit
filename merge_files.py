import os
import pandas as pd
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





CLIENT_ID = "c0510a4d3251459f900ddd4f78026045"
CLIENT_SECRET = "7afb92177f4342cd804e1e57130a23b8"
my_username = "westcoastmel"

# instantiating the client.  This 'sp' version of the client is used repeatedly below
# source: Max Hilsdorf (https://towardsdatascience.com/how-to-create-large-music-datasets-using-spotipy-40e7242cc6a6)
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, backoff_factor=2)


AUTH_URL = 'https://accounts.spotify.com/api/token'

# POST
auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})

# convert the response to JSON
auth_response_data = auth_response.json()

# save the access token
access_token = auth_response_data['access_token']

def get_audio_features_slowly(playlist_tracks, time_delay, sp):
    track_info = playlist_tracks.apply(lambda row: row["items"]["track"], axis=1).to_list()
    track_dict_list = []
    for track in track_info:
        try:
            time.sleep(time_delay)
            this_track_dict = {
                'track_id' : track['id'],
                'track_title' : track['name'],
                'artist_name' : track['artists'][0]['name']}
            audio_features_temp = sp.audio_features(track['id'])[0]
            # test for missing values
            this_track_dict.update(audio_features_temp)
            track_dict_list.append(this_track_dict)
        except Exception as e:
            print(e, track['id'])
    audio_features = pd.DataFrame(track_dict_list)
    return audio_features

def link_to_id(link):
    trimmed = link[34:]
    ans = trimmed.split('?')
    return ans[0]


def gen_csv(link, email, path):
    playlist_id_wr = link_to_id(link)
    name = email

    playlist_tracks_wr = pd.DataFrame(sp.user_playlist_tracks("westcoastmel", playlist_id_wr))
    
    feature_data = get_audio_features_slowly(playlist_tracks_wr, 2, sp)
    
    name_list = []
    
    
    for i in range(100):
        name_list.append(name)
    
    feature_data.insert(1, "Email", name_list)
    
                             
    feature_data.to_csv(path + '/spotify'+name+'.csv', index=False)
    











path = ''

folder_name = path + 'spotify_files'  # Replace with the actual folder path

def merge_csv_files(path, folder_name):
    


    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_name) if file.endswith('.csv')]

    # Initialize an empty DataFrame
    merged_df = pd.DataFrame()

    # Iterate over each CSV file and vertically merge them
    for file in csv_files:
        file_path = os.path.join(folder_name, file)
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], axis=0)

    # Reset the index of the merged DataFrame
    merged_df.reset_index(drop=True, inplace=True)

    # Write merged_df to a CSV file
    merged_df.to_csv(path + 'data.csv', index=False)


