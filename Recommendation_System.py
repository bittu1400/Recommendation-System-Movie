#!/usr/bin/env python
# coding: utf-8

# In[41]:


# get_ipython().system('pip install numpy pandas scikit-learn wordcloud seaborn matplotlib faiss-cpu nltk ')
# get_ipython().system('pip install --upgrade wordcloud pillow')


# In[2]:


# get_ipython().system('echo "# Movie-recommendation-system" >> README.md')
# get_ipython().system('git init')
# get_ipython().system('git add README.md')
# get_ipython().system('git commit -m "first commit"')
# get_ipython().system('git branch -M main')
# get_ipython().system('git remote add origin https://github.com/bittu1400/Movie-recommendation-system.git')
# get_ipython().system('git push -u origin main')


# In[42]:


# Import libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')


# In[43]:


# # Download NLTK data
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')


# In[44]:


# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) | {
    'movie', 'film', 'story', 'director', 'cast', 'released', 'production',
    'is', 'am', 'are', 'was', 'were', 'should', 'could', 'would', 'the', 'a', 'an',
    'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'
}


# In[45]:


# Load and preprocess data
df = pd.read_csv('./CSVs/movie_dataset_cleaned.csv')


# In[46]:


df


# In[47]:


df.columns


# In[48]:


# Drop rows where all columns are null
df = df.dropna(how='all')


# In[49]:


# Select relevant columns
columns_to_keep = [
    'title', 'vote_average', 'vote_count', 'status', 'release_date',
    'revenue', 'runtime', 'adult', 'budget', 'original_language',
    'original_title', 'overview', 'popularity', 'tagline', 'genres',
    'production_companies', 'production_countries', 'spoken_languages', 'keywords'
]
df = df[columns_to_keep]


# In[50]:


# Drop rows with any null values
df = df.dropna()
# Drop approximately half of the rows with any null values
# null_rows = df[df[columns_to_keep].isnull().any(axis=1)]
# non_null_rows = df[~df[columns_to_keep].isnull().any(axis=1)]
# np.random.seed(42)  # For reproducibility
# drop_indices = np.random.choice(null_rows.index, size=int(len(null_rows) / 2), replace=False)
# df = pd.concat([non_null_rows, null_rows.drop(drop_indices)]).reset_index(drop=True)


# In[51]:


df


# In[52]:


# Preprocess text with lemmatization
def preprocess_text(text):
    if pd.isna(text):
        return ''  # Handle potential NaN values
    tokens = word_tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)


# In[53]:


# Combine textual columns for TF-IDF
textual_columns = [
    'title', 'overview', 'keywords', 'genres', 'tagline',
    'production_companies', 'production_countries', 'spoken_languages',
    'original_language', 'original_title', 'status'
]
df['text'] = df[textual_columns].astype(str).agg(' '.join, axis=1).apply(preprocess_text)


# In[94]:


# Clean Genres column for plotting
df['PrimaryGenre'] = df['genres'].str.split(',').str[0].str.strip()
df['PrimaryGenre'] = df['PrimaryGenre'].replace('', 'Unknown')
top_genres = df['PrimaryGenre'].value_counts().head(20).index
df['PrimaryGenre'] = df['PrimaryGenre'].where(df['PrimaryGenre'].isin(top_genres), 'Other')


# In[55]:


# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=9000)
tfidf_matrix = tfidf.fit_transform(df['text'])
feature_names = tfidf.get_feature_names_out()
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")


# In[56]:


# --- CONFIGURATION ---
N_NEIGHBORS = 10
TOP_N = 5
SVD_COMPONENTS = 4  # For visualization
SVD_REDUCTION_COMPONENTS = 100  # Reduced for clustering
SAMPLE_SIZE = 20000  # For silhouette evaluation
TUNING_SAMPLE_SIZE = 15000  # For HDBSCAN tuning
RANDOM_STATE = 42


# In[57]:


# Dimensionality reduction for clustering
print(f"Reducing TF-IDF matrix to {SVD_REDUCTION_COMPONENTS} dimensions...")
svd = TruncatedSVD(n_components=SVD_REDUCTION_COMPONENTS, random_state=RANDOM_STATE)
tfidf_reduced = svd.fit_transform(tfidf_matrix)
print(f"Reduced Matrix Shape: {tfidf_reduced.shape}")


# In[58]:


# Normalize features for cosine similarity
tfidf_reduced = tfidf_reduced / np.linalg.norm(tfidf_reduced, axis=1, keepdims=True)


# In[59]:


# FAISS Setup
d = tfidf_reduced.shape[1]
index = faiss.IndexFlatIP(d)
chunk_size = 10000
for i in range(0, tfidf_reduced.shape[0], chunk_size):
    chunk = tfidf_reduced[i:i + chunk_size].astype('float32')
    faiss.normalize_L2(chunk)
    index.add(chunk)
print(f"FAISS Index Built: {index.ntotal} items")


# In[60]:


def recommend_content(title: str, top_n: int = TOP_N):
    matched = df[df['title'].str.lower() == title.lower()]
    if matched.empty:
        return f"Movie '{title}' not found in dataset."
    idx = matched.index[0]
    query_vec = tfidf_reduced[idx:idx+1].astype('float32')
    faiss.normalize_L2(query_vec)
    distances, indices = index.search(query_vec, top_n + 1)
    neighbor_indices = indices[0][1:]
    similarity_scores = distances[0][1:]

    recommendations = df.iloc[neighbor_indices][[
        'title', 'PrimaryGenre', 'production_companies', 'overview',
        'popularity', 'tagline', 'keywords', 'revenue', 'runtime', 'vote_average'
    ]].copy()
    recommendations['Similarity Score'] = similarity_scores
    recommendations = recommendations.sort_values(by='Similarity Score', ascending=False)
    return recommendations.head(top_n)


# In[61]:


print("Tuning KMeans parameters on sample...")
n_clusters_list = [17, 18, 19, 20, 21, 22, 23, 24]
best_score = -1
best_n_clusters = 10


# In[62]:


np.random.seed(RANDOM_STATE)
sample_indices = np.random.choice(tfidf_reduced.shape[0], TUNING_SAMPLE_SIZE, replace=False)
sample_data = tfidf_reduced[sample_indices]


# In[63]:


# Precompute cosine distance matrix for sample
sample_distances = cosine_distances(sample_data)


# In[64]:


for n_clusters in n_clusters_list:
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(sample_data)
    if len(set(clusters)) > 1:
        score = silhouette_score(sample_data, clusters, metric='cosine', sample_size=TUNING_SAMPLE_SIZE, random_state=RANDOM_STATE)
        print(f"Silhouette Score for n_clusters={n_clusters}: {score:.3f}")
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters


# In[65]:


print(f"\nBest configuration: n_clusters={best_n_clusters}, Silhouette Score={best_score:.3f}")


# In[66]:


print("Clustering with KMeans on full data...")
kmeans = KMeans(n_clusters=best_n_clusters, random_state=RANDOM_STATE)
df['Cluster'] = kmeans.fit_predict(tfidf_reduced)


# In[67]:


# Silhouette Score
if len(set(df['Cluster'])) > 1:
    sil_score = silhouette_score(tfidf_reduced, df['Cluster'], metric='cosine', sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"Final Silhouette Score: {sil_score:.3f}")
else:
    print("Final Silhouette Score not computed: fewer than 2 clusters formed.")


# In[68]:


# --- CLUSTER SUMMARY ---
cluster_summary = df.groupby('Cluster').agg(
    Count=('title', 'count'),
    TopGenres=('PrimaryGenre', lambda x: x.value_counts().head(3).to_dict())
)
print("\nCluster Sizes and Top Genres:")
print(cluster_summary)


# In[69]:


# Sample Movies
print("\nSample Movies per Cluster (Top 3):")
for cluster, group in df.groupby('Cluster'):
    print(f"\nCluster {cluster}:")
    sample = group[['title', 'PrimaryGenre', 'overview']].head(3)
    for _, row in sample.iterrows():
        print(f"- {row['title']} ({row['PrimaryGenre']}): {row['overview'][:100]}...")


# In[70]:


if len(set(df['Cluster'])) > 1:
    silhouette_vals = silhouette_samples(tfidf_reduced, df['Cluster'], metric='cosine')
    plt.figure(figsize=(10, 6))
    y_lower = 0
    for cluster in sorted(set(df['Cluster'])):
        cluster_sil_vals = np.sort(silhouette_vals[df['Cluster'] == cluster])
        y_upper = y_lower + len(cluster_sil_vals)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_vals, alpha=0.7, label=f'Cluster {cluster}')
        y_lower = y_upper
    plt.axvline(sil_score, color='red', linestyle='--', label='Avg Silhouette Score')
    plt.title('Silhouette Plot for KMeans Clustering')
    plt.xlabel('Silhouette Coefficient')
    plt.ylabel('Cluster')
    plt.legend()
    plt.savefig('silhouette_plot.png')
    plt.show()


# In[71]:


# Truncated SVD Visualization
svd_viz = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
svd_result = svd_viz.fit_transform(tfidf_matrix)
plt.figure(figsize=(18, 8))
sns.scatterplot(x=svd_result[:, 0], y=svd_result[:, 1], hue=df['Cluster'], palette='Set1', legend='full')
plt.title('Movie Clusters (TruncatedSVD)')
plt.xlabel('SVD 1')
plt.ylabel('SVD 2')
plt.savefig('svd_visualization.png')
plt.show()


# In[72]:


# --- GENRE DISTRIBUTION ---
plt.figure(figsize=(26, 8))
sns.countplot(data=df, x='Cluster', hue='PrimaryGenre', palette='Set2')
plt.title('Genre Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Movies')
plt.legend(title='Primary Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('genre_distribution.png')
plt.show()


# In[73]:


# --- WORD CLOUDS PER CLUSTER ---
import os
print(os.path.exists('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
for cluster, group in df.groupby('Cluster'):
    cluster_text = ' '.join(group['text'].astype(str))
    wordcloud = WordCloud(
        width=800, height=400, background_color='white', max_words=50,
#         font_path='/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
    ).generate(cluster_text)
    plt.figure(figsize=(18, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for Cluster {cluster}')
    plt.axis('off')
    plt.savefig(f'wordcloud_cluster_{cluster}.png')
    plt.show()


# In[74]:


df


# In[84]:


import pandas as pd
import numpy as np

# Load the cleaned dataset
df = pd.read_csv('./CSVs/movie_dataset_cleaned.csv')

# Approximate sample size (since your system samples for clustering)
SAMPLE_SIZE = 15000  # or the size you used in your clustering sample

# If the dataset is smaller than SAMPLE_SIZE, just use the whole dataset
if len(df) < SAMPLE_SIZE:
    sampled_df = df
else:
    sampled_df = df.sample(SAMPLE_SIZE, random_state=42)

# Randomly select 5 movie titles from the sampled data
random_titles = sampled_df['title'].dropna().sample(5, random_state=42).tolist()

print("🎬 Here are 5 randomly selected movie titles:")
for i, title in enumerate(random_titles, 1):
    print(f"{i}. {title}")


# In[89]:


sample_movie = 'Night Moves'
print(f"\nValidating Recommendations for '{sample_movie}':")
if df['title'].str.lower().eq(sample_movie.lower()).any():
    movie_info = df[df['title'].str.lower() == sample_movie.lower()].iloc[0]
    print(f"Input Movie Details:\n- {movie_info['title']} (Genre: {movie_info['PrimaryGenre']}, Companies: {movie_info['production_companies']}): {movie_info['overview'][:100]}...")
    recs = recommend_content(sample_movie, top_n=TOP_N)
    if isinstance(recs, str):
        print(recs)
    else:
        print("\nRecommended Movies:")
        print(recs[['title', 'PrimaryGenre', 'production_companies', 'Similarity Score', 'overview', 'vote_average', 'popularity']].to_string(index=False))
    
        neighbor_indices = recs.index
        sim_scores = recs['Similarity Score']
        top_titles = recs['title'].values

        plt.figure(figsize=(10, 6))
        sns.barplot(x=sim_scores, y=top_titles, palette='Blues_r')
        plt.title(f'Similarity Scores for \"{sample_movie}\"')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Movie Title')
        plt.savefig('similarity_scores.png')
        plt.show()
else:
    print(f"Movie '{sample_movie}' not found in dataset.")


# In[77]:


df.to_csv('./CSVs/movie_dataset_cleaned.csv', index=False)


# In[78]:


import pandas as pd

# Replace 'your_dataset.csv' with the path to your CSV file
df = pd.read_csv('./CSVs/movie_dataset_cleaned.csv', nrows=8000)

# Now df contains only the first 8000 rows of your dataset
print(df.shape)  # Should print (8000, number_of_columns)

# If you want to save these 8000 rows to a new CSV:
df.to_csv('./CSVs/subset_8000_rows.csv', index=False)


# In[ ]:




