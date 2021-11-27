import torch
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import IPython
import os
import zipfile
import os
import torch.nn as nn
import plotly.express as px
import streamlit as st

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import optim as optim, functional as F
from torchvision.transforms import ToTensor
from torch.nn.utils.rnn import pad_sequence
from matplotlib import pyplot as plt, transforms

from sklearn.tree import export_graphviz
import pydot

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
from subprocess import check_call

dataset = pd.read_csv('water_potability.csv')

df = dataset.replace(to_replace=np.nan, value=0)
df.head()

def normalize(vals, iter):
    assert type(vals) is list
    mm = min(iter)
    mx = max(iter)

    return [(i-mm)/(mx-mm) for i in vals]


norm_df = pd.DataFrame()

for i in df.iloc[:-1]:
    list_data = df[i].to_list()
    avg = normalize([sum(list_data)/len(list_data)], list_data)
    norm_data = normalize(list_data, list_data)
    norm_df[i] = norm_data

norm_df.head()

# print(norm_df.to_string())

norm_df = norm_df.sample(frac=1).reset_index(drop=True)
norm_df.head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x_train, x_test, y_train, y_test = train_test_split(norm_df.iloc[:,:-1], norm_df.iloc[:,-1])

assert (len(x_train))==(len(y_train))
assert (len(x_test))==(len(y_test))

rfregression = RandomForestRegressor(n_estimators = 1000)
rfregression.fit(x_train, y_train)

preds = rfregression.predict(x_test)
errors = abs(preds-y_test)

error_log = []
y_test_log = []
error_log.append(np.mean(errors))
y_test_log.append(np.mean(y_test))

acc = 100 - np.mean(errors)/np.mean(y_test)
# print(f"Accuracy: {acc}%")


# print(list(norm_df.iloc[:, :-1]))
tree = rfregression.estimators_[5]
export_graphviz(tree, out_file = 'tree.dot', feature_names=list(norm_df.iloc[:, :-1]), precision=1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
importances = list(rfregression.feature_importances_)
feat_importance = [(norm_df, round(importance, 2)) for feature, importance in zip(list(norm_df.iloc[:, :-1]), importances)]
feat_importance = sorted(feat_importance, key=lambda x: x[1], reverse=True)

print(*[f"Name: {list(norm_df.iloc[:, :-1])[idx]}, Importance: {feat_importance[idx][-1]}" for idx in range(len(feat_importance))])


st.title("Posiedon Water Works")
st.markdown("Welcome to our attempt at making water quality transparent and easy to access for indigenous communities!")

st.header("Water Quality Metrics")
st.markdown("> We use key features like 'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', and 'Turbidity' to calculate the quality of water with our state-of-the-art hardware coupled with our global informatics system that allows communities to easily identify and access water quality readings and thus redirect water flow accordingly.'")

st.dataframe(dataset.head())
st.text(f"The dataset provided has a shape of {dataset.shape}")

flag_viz = False

def train_model(norm_df):
    x_train, x_test, y_train, y_test = train_test_split(norm_df.iloc[:,:-1], norm_df.iloc[:,-1])

    assert (len(x_train))==(len(y_train))
    assert (len(x_test))==(len(y_test))

    rfregression = RandomForestRegressor(n_estimators = 1000)
    st.text('Training.... ')
    rfregression.fit(x_train, y_train)

    preds = rfregression.predict(x_test)
    errors = abs(preds-y_test)

    error_log = []
    y_test_log = []
    error_log.append(np.mean(errors))
    y_test_log.append(np.mean(y_test))

    acc = 100 - np.mean(errors)/np.mean(y_test)
    st.markdown(f"Accuracy: {acc}%")

    flag_viz = True


    tree = rfregression.estimators_[5]
    export_graphviz(tree, out_file = 'tree.dot', feature_names=list(norm_df.iloc[:, :-1]), precision=1)
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    importances = list(rfregression.feature_importances_)
    feat_importance = [(norm_df, round(importance, 2)) for feature, importance in zip(list(norm_df.iloc[:, :-1]), importances)]
    feat_importance = sorted(feat_importance, key=lambda x: x[1], reverse=True)

    st.subheader("Visualization of the feature importance for a given model prediction.")
    all_feats = [f"{list(norm_df.iloc[:, :-1])[idx]}:{feat_importance[idx][-1]}\n" for idx in range(len(feat_importance))]
    st.markdown(' '.join(i for i in all_feats))

    check_call(['dot','-Tpng','tree.dot','-o','output.png'])
    st.image(Image.open("output.png"), caption='Random Forest Visualizer')

    return rfregression


if st.button('Train the model here!'):
    model = train_model(norm_df)

ph_st = st.sidebar.slider('pH on scale 0-14', min_value=float(df['ph'].min()), max_value=float(df['ph'].max()))
Hardness_st = st.sidebar.slider('Hardness: Capacity of water to precipitate soap in mg/L.', min_value=float(df['Hardness'].min()), max_value=float(df['Hardness'].max()))
Solids_st = st.sidebar.slider('Solid: Total dissolved solids in ppm', min_value=float(df['Solids'].min()), max_value=float(df['Solids'].max()))
Chloramines_st = st.sidebar.slider('Chloramines: Amount of Chloramines in ppm.', min_value=float(df['Chloramines'].min()), max_value=float(df['Chloramines'].max()))
Sulfate_st = st.sidebar.slider('Sulfate: Amount of Sulfates dissolved in mg/L.', min_value=float(df['Sulfate'].min()), max_value=float(df['Sulfate'].max()))
Conductivity_st = st.sidebar.slider('Conductivity: Electrical conductivity of water in μS/cm.', min_value=float(df['Conductivity'].min()), max_value=float(df['Conductivity'].max()))
Organic_carbon_st = st.sidebar.slider('Organic_carbon: Amount of organic carbon in ppm.', min_value=float(df['Organic_carbon'].min()), max_value=float(df['Organic_carbon'].max()))
Trihalomethanes_st = st.sidebar.slider('Trihalomethanes: Amount of Trihalomethanes in μg/L.', min_value=float(df['Trihalomethanes'].min()), max_value=float(df['Trihalomethanes'].max()))
Turbidity_st = st.sidebar.slider('Turbidity: Measure of light emiting property of water in NTU.', min_value=float(df['Turbidity'].min()), max_value=float(df['Turbidity'].max()))

st.header('Make a custom prediction on arduino-collected water-quality data.')

if st.button('Make prediction on your inputted data!'):
    pred = np.array([ph_st, Hardness_st, Solids_st, Chloramines_st, Sulfate_st, Conductivity_st, Organic_carbon_st, Trihalomethanes_st, Turbidity_st])
    pred = np.expand_dims(pred, axis=0)    
    st.text(pred)
    rfreg_pred = rfregression.predict(pred)
    st.subheader(f"The estimated quality of your water sample is {rfreg_pred}")

    if rfreg_pred < 0.3:
        st.markdown('Careful. The water quality tested is unsafe to drink')
    elif 0.3 < rfreg_pred < 0.7:
        st.markdown(f'The water must be boiled for it to be safe to drink.')
    else:
        st.markdown('The water is primarily safe to drink.')