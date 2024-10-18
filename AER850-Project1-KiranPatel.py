#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# By: Kiran Patel
# AER 850 - Introduction to Machine Learning 
# Project 1
# Due: Oct 20th 2024
# 

"""
Created on Thu Oct 10 12:59:41 2024

@author: kiranpatel
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


#Step 1: Read Data from CSV file
# df = pd.read_csv converts the csv datafile into a dataframe
# This will allow for further analysis and data manipulation
df = pd.read_csv("AER850 - Project_1_Data.csv");


# Step 2: Data Visualization

# Create a figure and a 3D axis to plot the data
# A 3d Plot is utilized as the data consists of X,Y,Z coordinates 
# (i.e) 
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')

# Create a 3D scatter plot with color-coded points based on "Step"
scatterplot = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='viridis', marker='o')
colourbar = plt.colorbar(scatterplot,orientation='vertical', cax=fig1.add_axes([0.85, 0.15, 0.03, 0.7]))
colourbar.set_label('Step')

# Axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the 3D plot in the plotting panel
plt.show()


#Step 3 - Correlation Analysis
# A Heat map is used to make a Correlation Matrix to see if any points of data 
# have strong correlations and can be removed from analysis 

correlation_matrix = df.corr()
sbn.heatmap(np.abs(correlation_matrix))



# Step 4 - Classification Model Development/Engineering 











