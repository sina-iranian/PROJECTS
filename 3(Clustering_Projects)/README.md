# Clustering Project Placeholder
🌸 Clustering the Iris Dataset using PCA, DBSCAN, and KMeans

This project applies unsupervised learning techniques on the classic Iris dataset to explore natural groupings of flowers based on their sepal and petal dimensions. It combines Principal Component Analysis (PCA) for dimensionality reduction, DBSCAN for density-based clustering, and KMeans for center-based clustering refinement. The results are visualized using t-SNE, and various clustering performance metrics are calculated to evaluate the models.

📁 Dataset Used

iris_dataset.csv: Contains four numerical features — sepal length, sepal width, petal length, petal width — without labels (unsupervised setting).

📌 Project Workflow
🔹 Step 1: Data Preprocessing

Loaded the dataset using pandas.

Selected sepal and petal features separately.

Handled missing values using fillna(0).

🔹 Step 2: Feature Scaling & PCA

Applied StandardScaler to normalize features.

Ran PCA separately on sepal and petal features (2 components each).

Combined the 4 resulting PCA components into one feature set for clustering.

🔹 Step 3: DBSCAN Clustering

Ran DBSCAN on the combined PCA features (eps=0.5, min_samples=4).

Assigned DBSCAN labels to the original data.

Visualized clusters using t-SNE for better separation in 2D space.

🔹 Step 4: DBSCAN Hyperparameter Tuning

Evaluated DBSCAN performance across a range of eps and min_samples values.

Calculated clustering evaluation metrics:

Silhouette Score

Davies-Bouldin Index

Calinski-Harabasz Index

Helped identify optimal eps/min_samples combinations.

🔹 Step 5: KMeans Clustering (on DBSCAN Filtered Data)

Filtered out DBSCAN noise points (label = -1).

Applied KMeans (k=3) on the cleaned PCA data.

Visualized KMeans clusters using t-SNE, only for valid DBSCAN points.

Evaluated final KMeans clustering using:

Silhouette Score: 0.490

Davies-Bouldin Index: 0.823

Calinski-Harabasz Index: 284.411

📊 Key Insights

PCA helped preserve variance while reducing dimensionality for better clustering.

DBSCAN was effective at identifying noise and dense clusters without needing to predefine k.

KMeans further refined clusters on the DBSCAN-filtered dataset.

The pipeline demonstrates a hybrid clustering strategy, combining the strengths of both density-based and centroid-based methods.

🛠️ Technologies Used

pandas, numpy – Data handling

scikit-learn – PCA, DBSCAN, KMeans, evaluation metrics

seaborn, matplotlib – Visualizations

t-SNE – Dimensionality reduction for visualization
