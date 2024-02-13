import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
data = pd.read_csv("Medicine_Details.csv")

# Preprocess the data
# Assuming the dataset is already cleaned and encoded properly

# Feature Engineering
# Concatenate symptom and side effects columns to create a corpus
data["Symptom_SideEffects"] = data["Uses"] + " " + data["Side_effects"]

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

# Fit and transform the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(data["Symptom_SideEffects"])

# Compute similarity scores using linear kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Function to recommend medicines based on symptoms
def recommend_medicines(symptoms, cosine_sim=cosine_sim):
    # Transform input symptoms to TF-IDF vector
    symptoms_tfidf = tfidf_vectorizer.transform([symptoms])

    # Calculate similarity scores
    cosine_scores = linear_kernel(symptoms_tfidf, tfidf_matrix).flatten()

    # Get indices of top 5 medicines with highest similarity scores
    top_indices = cosine_scores.argsort()[:-6:-1]

    # Get the names of top 5 recommended medicines
    top_medicines = data.iloc[top_indices]["Medicine Name"].values

    return top_medicines


# Example usage
symptoms_input = "headache nausea"
recommended_medicines = recommend_medicines(symptoms_input)
print("Recommended Medicines:")
count = 0
for med in recommended_medicines:
    print("-", med)
    count += 1
    if count == 5:
        break
