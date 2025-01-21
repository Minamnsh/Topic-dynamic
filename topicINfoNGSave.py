import pandas as pd
import numpy as np
import joblib  # Import joblib for saving and loading models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# Load German and English stopwords
german_stopwords = stopwords.words('german')
english_stopwords = stopwords.words('english')

# Add specific words to stopwords
custom_stopwords = [
    'projekt', 'inwieweit', 'tp', 'viele', 'mehr', 'immer', 'innen', 'autor', 
    'literarischen', 'literarischer', 'literarische', 'feldes', 'liszt', 'konnte', 
    'drei', 'geeignetste', 'reich', 'zweck', 'insbesondere', 'better', 'often',
    'projekts', 'aufgrund', 'dabei','ermöglichen', 'prozent', 'sowie', 'neue', 
    'beide', 'neuer', 'wurde','fur', 'sollen', 'jedoch'
]

# Combine all stopwords
all_stopwords = list(set(german_stopwords + english_stopwords + custom_stopwords))

# Load data from Excel file
data = pd.read_excel('D:\\uni hildesheim\\MasterArbeit\\Gepris\\gepris-crawler-master\\InformatikFinal2023DA.xlsx')

# Fill NaN values in the 'description' column with an empty string
data['description'] = data['description'].fillna('')

# Remove "-" when it's in the middle of the word
data['description'] = data['description'].str.replace(r'\b-\b', '', regex=True)

# Define the number of topics
num_topics = 13

# Tokenizer for German text
tokenizer = RegexpTokenizer(r'\b\w\w+\b')

# Vectorize the text data, including unigrams and bigrams
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=all_stopwords, tokenizer=tokenizer.tokenize, ngram_range=(1, 2))
tfidf = tfidf_vectorizer.fit_transform(data['description'])

# Fit the NMF model
nmf_model = NMF(n_components=num_topics, random_state=42)
nmf_model.fit(tfidf)

# Save the trained NMF model and the TF-IDF vectorizer using joblib
joblib.dump(nmf_model, 'nmf_modelInfo.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizerInfo.pkl')

# Function to display top words for each topic
def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic %d:" % (topic_idx + 1))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

# Display the top words for each topic
num_top_words = 10
feature_names = tfidf_vectorizer.get_feature_names_out()
display_topics(nmf_model, feature_names, num_top_words)

# Function to plot histograms for topics
def plot_topic_histograms(model, feature_names, num_top_words):
    fig, axes = plt.subplots(5, 4, figsize=(11.69, 16.54), sharex=True)  # A4 dimensions in inches
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx < len(axes):
            top_features_ind = topic.argsort()[:-num_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.4)  # Increased bar height for more spacing
            ax.set_title(f'Topic {topic_idx + 1}', fontsize=8, pad=0)  # Adjusted title size and padding
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='major', labelsize=7)  # Adjusted label size
            for i in 'top right left'.split():
                ax.spines[i].set_visible(False)
    # Hide any unused subplots
    for i in range(num_topics, len(axes)):
        fig.delaxes(axes[i])
    fig.subplots_adjust(wspace=0.6, hspace=1.2)  # Adjusted space between plots
    plt.tight_layout(pad=2.0)
    plt.show()

# Plot histograms for topics
plot_topic_histograms(nmf_model, feature_names, num_top_words)
