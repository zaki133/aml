import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories for saving the plots
os.makedirs("boxplots", exist_ok=True)
os.makedirs("histograms", exist_ok=True)

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v5_unscaled.csv')

# Identify numerical features
numerical_features = df.select_dtypes(include=[np.number])

# Determine the number of features and how many plots per figure
num_features = len(numerical_features.columns)
plots_per_figure = 36
num_figures = (num_features + plots_per_figure - 1) // plots_per_figure

# Plot box plots for numerical features
for fig_num in range(num_figures):
    plt.figure(figsize=(20, 15))
    start_col = fig_num * plots_per_figure
    end_col = min(start_col + plots_per_figure, num_features)
    for i, col in enumerate(numerical_features.columns[start_col:end_col]):
        plt.subplot(6, 6, i+1)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.tight_layout()
    plt.savefig(f'boxplots/boxplot_{fig_num + 1}.png')
    plt.close()

# Plot histograms for numerical features
for fig_num in range(num_figures):
    plt.figure(figsize=(20, 15))
    start_col = fig_num * plots_per_figure
    end_col = min(start_col + plots_per_figure, num_features)
    for i, col in enumerate(numerical_features.columns[start_col:end_col]):
        plt.subplot(6, 6, i+1)
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.savefig(f'histograms/histogram_{fig_num + 1}.png')
    plt.close()
