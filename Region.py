# Import necessary libraries
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# File paths to load the pre-trained model and vectorizer (use the previously saved model)
model_path = 'nmf_modelInfo.pkl'
vectorizer_path = 'tfidf_vectorizerInfo.pkl'

# Load data from Excel file
data = pd.read_excel(r'D:\\uni hildesheim\\MasterArbeit\\Gepris\\gepris-crawler-master\\InformatikFinal2023DA.xlsx')

# Fill NaN values in the 'description' column with an empty string
data['description'] = data['description'].fillna('')

# Ensure that all Bundesländer are considered by replacing missing values with 'Unknown' or another placeholder
data['Bundesländer'] = data['Bundesländer'].fillna('Unknown')

# Load the pre-trained model and vectorizer (trained on the entire dataset)
nmf_model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(vectorizer_path)

# Apply the pre-trained model to the entire dataset
tfidf_data = tfidf_vectorizer.transform(data['description'])
W_data = nmf_model.transform(tfidf_data)

# Add topic distributions back to the DataFrame
num_topics = nmf_model.n_components_
topic_columns = [f'Topic_{i+1}' for i in range(num_topics)]
W_df = pd.DataFrame(W_data, columns=topic_columns)
data = pd.concat([data, W_df], axis=1)

# Calculate the centroid (average topic weight) for each Bundesland
centroid_by_bundesland = data.groupby('Bundesländer')[topic_columns].mean()

# Display the numerical values of the topic centroids by Bundesland in the terminal
print("\nNumerical Values of Topic Centroids by Federal States for All Researchers:")
print(centroid_by_bundesland)

# Function to display top words for each topic
def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

# Extract the top words for each topic and display them in the terminal
num_top_words = 10
feature_names = tfidf_vectorizer.get_feature_names_out()
display_topics(nmf_model, feature_names, num_top_words)

# Generate a heatmap of the topic centroids by Bundesland (topics on x-axis, Bundesländer on y-axis)
def plot_bundesland_centroid_heatmap(centroid_by_bundesland):
    plt.figure(figsize=(16, 10))  # Figure size for better fit
    sns.heatmap(centroid_by_bundesland, cmap="YlGnBu", cbar=True, annot=True, fmt=".2f", linewidths=0.5,
                cbar_kws={'label': 'Average Topic Weight'}, vmin=0, vmax=centroid_by_bundesland.values.max()-0.04,
                annot_kws={"size": 10},  # Adjust font size of annotations
                xticklabels=range(1, num_topics + 1),  # Use numbers 1, 2, 3,... as x-axis labels
                yticklabels=centroid_by_bundesland.index)  # Ensure Bundesländer are displayed as y-axis labels

    plt.title("Heatmap of Topic Centroids by Federal States for All Researchers", fontsize=16)  # Title with larger font size
    plt.xlabel("Topics", fontsize=14)  # X-axis label with larger font size
    plt.ylabel("Federal States", fontsize=14)  # Y-axis label with larger font size
    plt.xticks(rotation=0, ha='right', fontsize=12)  # Rotate x-axis labels, set font size
    plt.yticks(rotation=0, fontsize=12)  # Keep y-axis labels horizontal, set font size
    plt.tight_layout()  # Adjust layout to fit all elements
    plt.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.2)  # Adjust margins to fix layout
    plt.show()

# Plot the heatmap of centroids
plot_bundesland_centroid_heatmap(centroid_by_bundesland)
