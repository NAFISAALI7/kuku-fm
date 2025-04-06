# Install dependencies
# pip install scikit-learn nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Sample movie dataset
movies = [
    {"title": "The Office", "description": "A mockumentary sitcom about office life."},
    {"title": "Stranger Things", "description": "Sci-fi horror drama with kids and supernatural events."},
    {"title": "Brooklyn Nine-Nine", "description": "Comedy series set in a New York police department."},
    {"title": "Dark", "description": "German sci-fi series exploring time travel and mystery."},
    {"title": "Friends", "description": "Comedy series about six friends living in New York."}
]

# Preprocess movie descriptions
descriptions = [movie['description'] for movie in movies]
vectorizer = TfidfVectorizer().fit_transform(descriptions)
similarity_matrix = cosine_similarity(vectorizer)

# Simple chatbot
def recommend_movies(user_input):
    input_vec = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(input_vec, vectorizer).flatten()
    sorted_indices = similarity_scores.argsort()[::-1]
    
    print("\n🎥 Recommended Series:")
    for idx in sorted_indices[:3]:
        print(f"- {movies[idx]['title']}: {movies[idx]['description']}")

# Chatbot loop
print("🤖 Welcome to the Kuku FM Recommender Chatbot!")
while True:
    query = input("\nTell me what kind of shows you like (or type 'exit'): ")
    if query.lower() == 'exit':
        break
    recommend_movies(query)
