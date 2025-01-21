import pandas as pd
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk

# Ensure you have the necessary NLTK resources
nltk.download('stopwords')

# Combine with gensim and nltk stopwords
german_stopwords = stopwords.words('german')
english_stopwords = stopwords.words('english')
custom_stopwords = [
    'projekt', 'inwieweit', 'tp', 'viele', 'mehr', 'immer', 'innen', 'autor', 
    'literarischen', 'literarischer', 'literarische', 'feldes', 'liszt', 'konnte', 
    'drei', 'geeignetste', 'reich', 'zweck', 'insbesondere', 'better', 'often',
    'projekts', 'aufgrund', 'dabei','ermöglichen', 'prozent', 'sowie', 'neue', 
    'beide', 'neuer', 'wurde','fur', 'sollen', 'jedoch'
] # Add your custom stopwords here
all_stopwords = set(german_stopwords + english_stopwords + list(gensim.parsing.preprocessing.STOPWORDS) + custom_stopwords)

# Tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# Preprocessing function
def preprocess(text):
    if isinstance(text, str):
        return [word for word in gensim.utils.simple_preprocess(text, deacc=True) if word not in all_stopwords]
    else:
        return []

def main():
    # Load the Excel file
    file_path = 'D:\\uni hildesheim\\MasterArbeit\\Gepris\\gepris-crawler-master\\InformatikFinal2023DA.xlsx'  # Ensure the correct path
    df = pd.read_excel(file_path)

    # Preprocess the text data
    df['processed_description'] = df['description'].apply(preprocess)

    # Filter out empty documents
    df = df[df['processed_description'].str.len() > 0]

    # Create a dictionary and corpus for coherence
    texts = df['processed_description'].tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Prepare TF-IDF matrix for NMF
    df['processed_description_joined'] = df['processed_description'].apply(lambda x: ' '.join(x))
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, tokenizer=tokenizer.tokenize, ngram_range=(1, 2))
    tfidf = tfidf_vectorizer.fit_transform(df['processed_description_joined'])

    # Calculate coherence score for different numbers of topics
    coherence_scores = []
    model_list = []
    topic_word_lists = []  # To store topics for later access
    
    for num_topics in range(1, 26):  # Change the range as needed
        # Initialize and fit NMF model
        nmf_model = NMF(n_components=num_topics, random_state=42, max_iter=500, init='nndsvda', solver='mu')
        nmf_model.fit(tfidf)

        # Get the top words for each topic
        topics = []
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
            topics.append(top_words)

        # Store topics for later
        topic_word_lists.append(topics)

        # Calculate coherence score using gensim's CoherenceModel
        coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(coherence_model.get_coherence())
        model_list.append(nmf_model)

    # Output coherence scores and the optimal number of topics
    print(f'Coherence scores: {coherence_scores}')
    optimal_num_topics = coherence_scores.index(max(coherence_scores)) + 1
    print(f'Optimal number of topics: {optimal_num_topics}')

    # Plot coherence scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 26), coherence_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Coherence Scores by Number of Topics')
    plt.grid(True)
    plt.show()

    # Display the topics for the optimal number of topics
    optimal_topics = topic_word_lists[optimal_num_topics - 1]
    for idx, topic in enumerate(optimal_topics, 1):
        print(f"\nTopic {idx}:")
        print(" ".join(topic))

if __name__ == '__main__':
    main()

