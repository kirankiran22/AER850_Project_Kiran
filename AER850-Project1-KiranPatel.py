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


#Step 1: Read Data from CSV file
# df = pd.read_csv converts the csv datafile into a dataframe
# This will allow for further analysis and data manipulation
df = pd.read_csv("Project 1 Data.csv");


# Step 2: Data Visualization

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a 3D scatter plot with color-coded points based on "Step"
sc = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Step'], cmap='viridis', marker='o')

# Add color bar
cbar = plt.colorbar(sc,orientation='vertical', cax=fig.add_axes([0.85, 0.15, 0.03, 0.7]))

cbar.set_label('Step')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the 3D plot
plt.show()







