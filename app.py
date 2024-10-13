from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Fetch dataset, initialize vectorizer and LSA here
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Apply Truncated SVD for LSA
lsa = TruncatedSVD(n_components=100, random_state=42)
X_reduced = lsa.fit_transform(X)

def search_engine(query):
    """
    Function to search for top 5 similar documents given a query
    Input: query (str)
    Output: documents (list), similarities (list), indices (list)
    """
    # Vectorize the query
    query_vec = vectorizer.transform([query])
    # Reduce dimensionality
    query_reduced = lsa.transform(query_vec)
    # Compute cosine similarities
    similarities = cosine_similarity(query_reduced, X_reduced)[0]
    # Get top 5 documents
    indices = np.argsort(similarities)[::-1][:5]
    top_docs = [documents[i] for i in indices]
    top_similarities = [similarities[i] for i in indices]
    return top_docs, top_similarities, indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    # Convert similarities to floats for JSON serialization
    similarities = [float(s) for s in similarities]
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
