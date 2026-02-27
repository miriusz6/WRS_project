The project was created by me as examination sumbission for the Web Recommender System during my Master's studies. For the full pdf repport see 'WRS_Project.pdf'

## 1. Data Processing and Exploration

The initial dataset contained 14,000 entries consisting of user IDs, item IDs, ratings (1–5), and timestamps.

-   **Cleaning:** To ensure data quality,  duplicate entries were removed and filtered out items in the test set that did not appear in the training set.

-   **Analysis:** Exploration revealed that the data was highly **sparse** (only 2.44% of the possible rating matrix was filled) and heavily **skewed**, with approximately 88.7% of ratings being 4s or 5s.
    
-   **Long-Tail Distribution:** Both user activity and item popularity followed a "long tail" distribution, meaning a small minority of users and items accounted for the bulk of the interactions.
    

## 2. Recommendation Models

Three distinct categories of recommendation strategies were implemented:

### Collaborative Filtering (CF)

These models predict user preferences based on past behavior and similarities between users or items:

-   **TopPop:** A baseline model that recommends the most popular items to all users.
    
-   **KNN-WithMeans:** A K-Nearest Neighbors approach that accounts for the average rating of each user to find similar neighbors.
    
-   **SVD (Singular Value Decomposition):** A latent factor model used to uncover hidden patterns in user-item interactions.
    
    

### Content-Based (CB) Models

These models recommend items similar to those a user has liked in the past, based on item metadata like **titles** and **descriptions**:


-   **Text Representation:** Natural language processing (tokenization, lemmatization, and stop-word removal) was used to clean metadata.
    
-   **Embeddings:** Items were converted into numerical vectors using **TF-IDF** (statistical importance) and **Word2Vec** (semantic similarity).
    
-   **User Profiling:** User profiles were created by averaging the vectors of items they rated highly (above 3).
    
    

### Hybrid Systems

To leverage the strengths of both CF and CB methods, three hybrid strategies were tested:


-   **Pipeline Model:** Used the output of one model as the input for another (specifically using CB similarity values to train a KNN model).
    
-   **Parallel Model:** Re-ranked KNN recommendations by multiplying them with content-based similarity scores.
    
-   **Switching Strategy:** Chose between KNN and CB recommendations based on the user's activity level (e.g., using KNN for less active users and CB for more active ones).
    

## 3. Evaluation Strategy

The models were evaluated using two types of metrics to understand different aspects of performance:


-   **Error-Based:**  **RMSE** (Root Mean Square Error) was used to measure the accuracy of predicted ratings, specifically to penalize larger outliers.
    
-   **Rank-Based:** To evaluate the quality of "Top 10" recommendations, the project used **HitRate**, **Precision**, **MAP** (Mean Average Precision), **MRR** (Mean Reciprocal Rank), and **Coverage** (percentage of total items recommended).
