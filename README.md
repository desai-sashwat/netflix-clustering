# Netflix Show Clustering Using K-Means Algorithm

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
  - [Cluster Profiles](#cluster-profiles)
  - [Performance Metrics](#performance-metrics)
  - [Business Insights](#business-insights)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)

## Overview
This project implements an unsupervised machine learning approach to cluster Netflix content into meaningful groups using K-means clustering. By analyzing 8,807 movies and TV shows from the Netflix catalog, the system identifies content patterns and creates a recommendation engine based on similarity metrics. The project employs extensive feature engineering, combining numerical, categorical, and text-based features to capture the multi-dimensional nature of entertainment content.

## Dataset
The dataset used in this project consists of Netflix movies and TV shows with the following characteristics:

- **Source**: Netflix Movies and TV Shows dataset from Kaggle
- **Total Entries**: 8,807 titles
- **Content Types**: Movies (69.6%) and TV Shows (30.4%)
- **Features**: 12 original attributes including:
  - Title, Type, Director, Cast
  - Country, Date Added, Release Year
  - Rating, Duration
  - Listed In (Genres)
  - Description

### Data Processing
- **Missing Values**: Handled through strategic imputation for director (30.2% missing), cast (9.2% missing), country (9.7% missing), and rating fields
- **Feature Engineering**: Created 106 features across 7 categories for comprehensive content representation
- **Text Processing**: Applied TF-IDF vectorization on descriptions to extract 30 key terms

## Repository Structure
- **code/** - Implementation notebooks
  - **netflix_clustering.ipynb** - Main clustering implementation with EDA, feature engineering, and model development
  - **placeholder.md** - Placeholder file
- **data/** - Dataset directory
  - **netflix_titles.csv** - Source Netflix catalog dataset (3.4 MB)
  - **placeholder.md** - Placeholder file
- **outputs/** - Results and analysis outputs
  - **netflix_clustering_results.csv** - Complete dataset with cluster assignments (4.1 MB)
  - **netflix_clustering_summary.csv** - Cluster summary statistics
  - **placeholder.md** - Placeholder file
- **.gitignore** - Git ignore file for Python projects
- **LICENSE** - MIT License
- **README.md** - Project documentation
- **requirements.txt** - Python package dependencies

## Methodology
The clustering approach consists of multiple stages:

### 1. Data Preprocessing
- Imputation of missing values using domain-appropriate strategies
- Temporal feature extraction from date fields
- Standardization of duration metrics across movies and TV shows
- Creation of derived features (cast size, number of genres, country count)

### 2. Feature Engineering
Comprehensive feature extraction creating 106 total features:

| Feature Category | Count | Description |
|-----------------|-------|-------------|
| Numerical | 7 | Release year, duration, year/month added, cast size, etc. |
| Binary | 1 | Director availability indicator |
| Type Encoding | 2 | Movie vs TV Show one-hot encoding |
| Rating Encoding | 9 | Content rating categories |
| Genre Multi-label | 42 | Binary encoding for all unique genres |
| Country Encoding | 16 | Top 15 countries plus "Other" category |
| Text (TF-IDF) | 30 | Key terms from descriptions |

### 3. Model Development
- **Algorithm**: K-Means clustering with StandardScaler normalization
- **Optimal K Selection**: Evaluated K values from 2 to 15 using:
  - Elbow Method (Inertia analysis)
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Score
- **Final Configuration**: K=2 based on Silhouette Score optimization

### 4. Recommendation System
Content-based recommendation engine that:
- Identifies similar content within the same cluster
- Calculates Euclidean distance in feature space
- Returns top-N recommendations based on similarity scores

## Key Findings

### Cluster Profiles

| Cluster | Size | Percentage | Primary Type | Top Genre | Top Rating | Avg Release Year |
|---------|------|------------|--------------|-----------|------------|------------------|
| **Cluster 0** | 2,675 | 30.4% | TV Show | International TV Shows | TV-MA | 2016 |
| **Cluster 1** | 6,132 | 69.6% | Movie | Dramas | TV-MA | 2013 |

### Performance Metrics
- **Silhouette Score**: 0.2145 (indicating moderate cluster separation)
- **Davies-Bouldin Score**: 1.8932 (lower values indicate better clustering)
- **Calinski-Harabasz Score**: 1847.53 (higher values indicate better-defined clusters)
- **Inertia**: Optimal elbow point observed at K=2

### Business Insights
1. **Content Distribution**: Netflix catalog shows a 70-30 split between mainstream movie content and specialized TV show content
2. **Rating Patterns**: TV-MA rated content dominates both clusters, indicating adult-oriented programming strategy
3. **Temporal Trends**: Recent content (2013-2016 average) forms the majority of the catalog
4. **Geographic Diversity**: International content plays a significant role in content clustering
5. **Genre Concentration**: Drama and International content are primary differentiators between clusters

## Usage
### Prerequisites
```bash
# Clone the repository
git clone https://github.com/desai-sashwat/netflix-clustering.git
cd netflix-clustering

# Install required packages
pip install -r requirements.txt
```

### Running the Analysis
```python
# Open and run the Jupyter notebook
jupyter notebook code/netflix_clustering.ipynb
```

### Using the Recommendation System
```python
# Example: Get recommendations for a specific title
recommendations = get_recommendations(
    title_query="Stranger Things",
    netflix=netflix_clean,
    X_scaled=X_scaled,
    kmeans_model=kmeans_final,
    n_recommendations=5
)
```

### Output Files
- **netflix_clustering_results.csv**: Full dataset with cluster assignments for each title
- **netflix_clustering_summary.csv**: Aggregate statistics for each cluster

## Future Work
- **Multi-level Clustering**: Implement hierarchical clustering to identify sub-clusters within main groups
- **Deep Learning Integration**: Explore neural network embeddings for content representation
- **Temporal Analysis**: Study how cluster compositions change over time
- **Cross-platform Analysis**: Extend methodology to compare Netflix with other streaming platforms
- **User Integration**: Incorporate user ratings and viewing patterns for collaborative filtering
- **Real-time Updates**: Develop pipeline for continuous model updates as new content is added
- **Interpretability Enhancement**: Implement SHAP values for better feature importance understanding

## License
This project is licensed under the MIT License - see the LICENSE file for details.
