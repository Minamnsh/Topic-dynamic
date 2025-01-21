# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
import joblib
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import seaborn as sns

# File paths to load the pre-trained model and vectorizer (use the previously saved model)
model_path = 'nmf_modelInfo.pkl'
vectorizer_path = 'tfidf_vectorizerInfo.pkl'


# Load pre-trained NMF model and TfidfVectorizer using joblib
nmf_model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(vectorizer_path)

# Function to display topics in the terminal
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-no_top_words - 1:-1]])
        print(message)
    print("\n")

# Display the topics with the top 10 words in the terminal
no_top_words = 10
feature_names = tfidf_vectorizer.get_feature_names_out()
display_topics(nmf_model, feature_names, no_top_words)

# Load German and English stopwords
german_stopwords = stopwords.words('german')
english_stopwords = stopwords.words('english')
custom_stopwords = [
    'projekt', 'inwieweit', 'tp', 'viele', 'mehr', 'immer', 'innen', 'autor', 
    'literarischen', 'literarischer', 'literarische', 'feldes', 'liszt', 'konnte', 
    'drei', 'geeignetste', 'reich', 'zweck', 'insbesondere', 'better', 'often',
    'projekts', 'aufgrund', 'dabei','ermöglichen', 'prozent', 'sowie', 'neue', 
    'beide', 'neuer', 'wurde','fur', 'sollen', 'jedoch'
]
all_stopwords = list(set(german_stopwords + english_stopwords + custom_stopwords))

# Load data from Excel file
data = pd.read_excel(r'D:\\uni hildesheim\\MasterArbeit\\Gepris\\gepris-crawler-master\\InformatikFinal2023DA.xlsx')

# Fill NaN values in the 'description' and 'startYear' columns
data['description'] = data['description'].fillna('')
data['startYear'] = data['startYear'].fillna(0).astype(int)

# Filter data for 'Federal states' and make a copy to avoid SettingWithCopyWarning
city_data = data[data['Bundesländer'] == 'Baden-Württemberg'].copy()

# Remove "-" when it's in the middle of the word
city_data['description'] = city_data['description'].str.replace(r'\b-\b', '', regex=True)

# Vectorize the text data using the pre-trained TfidfVectorizer
tfidf = tfidf_vectorizer.transform(city_data['description'])

# Transform the text data using the pre-trained NMF model
nmf_topic_matrix = nmf_model.transform(tfidf)

# Add topic distributions back to the DataFrame
num_topics = nmf_model.n_components_
topic_columns = [f'Topic_{i+1}' for i in range(num_topics)]
city_data[topic_columns] = nmf_topic_matrix

# Group by startYear and calculate the mean topic weights for each year (centroid calculation)
topic_by_year = city_data.groupby('startYear')[topic_columns].mean()

# Calculate the differences in centroids between consecutive years
centroid_changes = topic_by_year.diff().fillna(0)

# Calculate the Euclidean distance of each centroid change to quantify overall movement
centroid_distance = np.linalg.norm(centroid_changes, axis=1)

# Set a minimum threshold to avoid division by small distances (e.g., 0.01)
min_threshold = 0.01
centroid_distance[centroid_distance < min_threshold] = np.nan

# Calculate the absolute contribution of each topic to the centroid change
contribution_abs = (centroid_changes.T / centroid_distance).abs().T

# Filter out very small changes (less than a certain threshold) to avoid noise
contribution_threshold = 0.01
contribution_abs[contribution_abs < contribution_threshold] = 0

# Normalize the contributions within each year so they sum up to 1
contribution_abs = contribution_abs.div(contribution_abs.sum(axis=1), axis=0).fillna(0)

# Exclude the first year from the plot (assuming first year index is 0)
contribution_abs_filtered = contribution_abs.iloc[1:]

# Create a heatmap to visualize the normalized absolute contribution of topics to the centroid changes
plt.figure(figsize=(14, 8))
sns.heatmap(contribution_abs_filtered.T, cmap='YlGnBu', annot=True, fmt=".2f")
plt.title('Heatmap of Topic Dynamics in Baden-Württemberg Over Time')
plt.xlabel('Year')
plt.ylabel('Topics')
plt.show()

# Apply t-SNE to reduce the dimensionality to 2D with a lower perplexity value (optional if needed)
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_results = tsne.fit_transform(topic_by_year.iloc[1:].values)  # Apply t-SNE to data without the first year

# Create a DataFrame for t-SNE results
trajectory_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
trajectory_df['Year'] = topic_by_year.index[1:]  # Exclude first year from the index as well
