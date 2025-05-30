# 🎬 Movie Recommendation System

A content-based movie recommendation system built with **Python**, leveraging **TF-IDF** vectorization, **TruncatedSVD** dimensionality reduction, **KMeans clustering**, and **FAISS** similarity search for fast and accurate recommendations.

This project uses a cleaned version of the **TMDB movie dataset**, featuring overviews, keywords, genres, production companies, and more, to generate recommendations based on textual and metadata similarity.

---

## 🚀 Features

✅ **Content-Based Filtering** – Generates recommendations by comparing textual metadata (title, overview, genres, keywords, etc.).
✅ **TF-IDF Vectorization** – Captures important terms from combined textual fields.
✅ **Dimensionality Reduction** – Reduces high-dimensional TF-IDF vectors using **TruncatedSVD** for efficient clustering.
✅ **KMeans Clustering** – Groups similar movies into clusters for better contextual recommendations.
✅ **FAISS Similarity Search** – Enables fast nearest-neighbor searches based on cosine similarity.
✅ **Visualization Tools** – Includes SVD plots, silhouette plots, genre distribution, and cluster-based word clouds.
✅ **Flexible Configurations** – Easily adjust TF-IDF features, SVD dimensions, KMeans clusters, and sampling sizes.

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/recommendation-system-movie.git
cd movie-recommendation-system
pip install -r requirements.txt
```

Required Python libraries:

* numpy
* pandas
* scikit-learn
* wordcloud
* seaborn
* matplotlib
* faiss-cpu
* nltk

Also, download necessary **NLTK data** (done automatically in the code):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 📂 Dataset

* **TMDB\_movie\_dataset\_v11.csv** – Raw dataset (used internally for cleaning).
* **cleaned\_movie\_dataset.csv** – Cleaned and preprocessed dataset.

Ensure the cleaned dataset (`cleaned_movie_dataset.csv`) is in the working directory.

---

## 🔎 Usage

Run the **Jupyter Notebook** or **Python script** to:

* Load and clean the dataset.
* Preprocess text and generate TF-IDF vectors.
* Apply **TruncatedSVD** for dimensionality reduction.
* Perform **KMeans clustering** for movie grouping.
* Build **FAISS index** for similarity search.
* Recommend similar movies for a given title:

```python
recommendations = recommend_content('Avatar', top_n=5)
print(recommendations)
```

Visualizations:

* **Silhouette Plots** for clustering quality.
* **SVD Scatter Plots** for visualizing clusters.
* **Word Clouds** for clusters.
* **Genre Distribution** across clusters.

---

## 🎥 Sample Workflow

1️⃣ Load the cleaned dataset (`cleaned_movie_dataset.csv`).
2️⃣ Preprocess and vectorize text using **TF-IDF**.
3️⃣ Reduce dimensions with **TruncatedSVD**.
4️⃣ Cluster movies with **KMeans**.
5️⃣ Build **FAISS** index and search for nearest neighbors.
6️⃣ Get recommendations with:

```python
recommend_content('Inception')
```

7️⃣ View **visualizations** generated in the notebook.

---

## ⚠️ Limitations

* Currently uses a **content-based approach** (no collaborative filtering or user history).
* Recommendations may be **cluster-biased** (e.g., overly focused on specific franchises like Batman).
* Dataset size is limited by sampling due to performance constraints.
* Diversity in recommendations can be improved.

---

## 💡 Future Improvements

* Integrate **collaborative filtering** or hybrid methods.
* Improve **genre weighting** and **similarity scoring**.
* Scale to **larger datasets** with optimized vectorization.
* Enhance visualizations and provide **interactive dashboards**.
* Add **real-time movie search interface**.

---

## 📄 License

This project is open source under the **MIT License**.

---

## 🤝 Contributing

Contributions are welcome! Please submit pull requests, issues, or suggestions.

---

## 🔗 Credits

* Inspired by the **TMDB** movie dataset.
* Built with **scikit-learn**, **faiss-cpu**, **nltk**, and **wordcloud**.