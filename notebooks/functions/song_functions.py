#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random
from sklearn.preprocessing import StandardScaler

from joblib import dump, load

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# In[2]:


spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id="a4cf71f1fda94d89849d2b479502cefc",
    client_secret="830c8872e4b3413bb51abafa8d0ccf14"))


# In[3]:


#import the data
playlist_songs = pd.read_csv("playlist_songs.csv")
hot100 = pd.read_csv("hot100songs.csv")


# In[4]:


kmeans = pickle.load(open("clustering_model", 'rb'))


# In[5]:


def get_audio_features(song, spotify):
    results = spotify.search(song, limit = 1)
    user_song_uri = results["tracks"]["items"][0]["uri"]
    user_song_audio_feat = spotify.audio_features(user_song_uri)
    return user_song_audio_feat


# In[6]:


def cluster_user_song(audio_features):
    scaler = load('std_scaler.bin')
    user_song_audio_feat_df = pd.DataFrame(audio_features)
    user_song_audio_feat_to_cluster = user_song_audio_feat_df[["danceability", "energy", "key", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]]
    user_song_audio_feat_scaled = scaler.transform(user_song_audio_feat_to_cluster)
    user_song_audio_feat_scaled = pd.DataFrame(user_song_audio_feat_scaled)
    cluster_user_song = kmeans.predict(user_song_audio_feat_scaled)
    return list(cluster_user_song)


# In[7]:


def suggesting_song(cluster):
    cluster_to_suggest = playlist_songs[playlist_songs["cluster"] == cluster[0]]
    song_to_suggest = cluster_to_suggest.sample(n=1)
    return song_to_suggest["song name"]


# In[ ]:


user_song = input("please select a song ")


# In[8]:


def hot_song(song):
    if hot100["song names"].str.contains(song).any():
        print("This is a hot song, this might like you: ", random.choice(hot100["song names"]))
    else:
        user_song_audio_feat = get_audio_features(user_song, spotify)
        user_song_cluster = cluster_user_song(user_song_audio_feat)
        song_to_suggest = suggesting_song(user_song_cluster)
        print("Try", list(song_to_suggest)[0], ", you might like it")


# In[ ]:




