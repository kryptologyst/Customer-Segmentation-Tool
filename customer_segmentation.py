"""
Enhanced Customer Segmentation Tool
Provides comprehensive customer segmentation using multiple algorithms and evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
    def __init__(self, data=None):
        self.data = data
        self.scaled_data = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.models = {}
        self.results = {}
        self.feature_columns = []
        
    def load_data(self, filepath=None, dataframe=None):
        """Load customer data from file or dataframe."""
        if dataframe is not None:
            self.data = dataframe.copy()
        elif filepath:
            self.data = pd.read_csv(filepath)
        else:
            raise ValueError("Either filepath or dataframe must be provided")
        
        print(f"Loaded data with {len(self.data)} customers and {len(self.data.columns)} features")
        return self.data
    
    def preprocess_data(self, features=None):
        """Preprocess data for clustering."""
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        # Select features for clustering
        if features is None:
            # Default features for clustering
            numeric_features = ['Age', 'AnnualIncome', 'SpendingScore', 'PurchaseFrequency', 
                              'AvgOrderValue', 'TotalSpent', 'MembershipYears']
            self.feature_columns = [col for col in numeric_features if col in self.data.columns]
        else:
            self.feature_columns = features
        
        # Handle categorical variables if needed
        categorical_features = ['Gender', 'Location']
        for cat_feature in categorical_features:
            if cat_feature in self.data.columns:
                le = LabelEncoder()
                self.data[f'{cat_feature}_encoded'] = le.fit_transform(self.data[cat_feature])
                if cat_feature not in self.feature_columns:
                    self.feature_columns.append(f'{cat_feature}_encoded')
        
        # Scale the features
        X = self.data[self.feature_columns]
        self.scaled_data = self.scaler.fit_transform(X)
        
        print(f"Preprocessed {len(self.feature_columns)} features: {self.feature_columns}")
        return self.scaled_data
    
    def find_optimal_clusters(self, max_clusters=10, method='kmeans'):
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        if self.scaled_data is None:
            raise ValueError("Data not preprocessed. Use preprocess_data() first.")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            if method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(self.scaled_data)
                inertias.append(model.inertia_)
            
            silhouette_avg = silhouette_score(self.scaled_data, labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find optimal k using silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k
        }
    
    def perform_clustering(self, algorithm='kmeans', n_clusters=4, **kwargs):
        """Perform clustering using specified algorithm."""
        if self.scaled_data is None:
            raise ValueError("Data not preprocessed. Use preprocess_data() first.")
        
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
        elif algorithm == 'dbscan':
            model = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 5))
        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        else:
            raise ValueError("Supported algorithms: 'kmeans', 'dbscan', 'hierarchical'")
        
        # Fit the model
        labels = model.fit_predict(self.scaled_data)
        
        # Store results
        self.models[algorithm] = model
        self.data[f'{algorithm}_cluster'] = labels
        
        # Calculate metrics
        if len(set(labels)) > 1:  # More than one cluster
            silhouette_avg = silhouette_score(self.scaled_data, labels)
            calinski_harabasz = calinski_harabasz_score(self.scaled_data, labels)
        else:
            silhouette_avg = -1
            calinski_harabasz = -1
        
        self.results[algorithm] = {
            'labels': labels,
            'n_clusters': len(set(labels)),
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz
        }
        
        print(f"{algorithm.upper()} Clustering Results:")
        print(f"Number of clusters: {len(set(labels))}")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
        
        return labels
    
    def analyze_segments(self, algorithm='kmeans'):
        """Analyze characteristics of each segment."""
        if algorithm not in self.results:
            raise ValueError(f"No results found for {algorithm}. Run perform_clustering() first.")
        
        cluster_col = f'{algorithm}_cluster'
        
        # Segment analysis
        segment_analysis = self.data.groupby(cluster_col).agg({
            'Age': ['mean', 'std', 'min', 'max'],
            'AnnualIncome': ['mean', 'std', 'min', 'max'],
            'SpendingScore': ['mean', 'std', 'min', 'max'],
            'PurchaseFrequency': ['mean', 'std'],
            'TotalSpent': ['mean', 'std'],
            'CustomerID': 'count'
        }).round(2)
        
        # Flatten column names
        segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns]
        segment_analysis = segment_analysis.rename(columns={'CustomerID_count': 'Count'})
        
        return segment_analysis
    
    def create_visualizations(self, algorithm='kmeans'):
        """Create comprehensive visualizations for the segmentation."""
        if algorithm not in self.results:
            raise ValueError(f"No results found for {algorithm}. Run perform_clustering() first.")
        
        cluster_col = f'{algorithm}_cluster'
        
        # PCA for 2D visualization
        pca_features = self.pca.fit_transform(self.scaled_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PCA Cluster Visualization', 'Age vs Income by Cluster',
                          'Spending Score Distribution', 'Cluster Size Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # PCA scatter plot
        for cluster in sorted(self.data[cluster_col].unique()):
            cluster_data = self.data[self.data[cluster_col] == cluster]
            cluster_pca = pca_features[self.data[cluster_col] == cluster]
            
            fig.add_trace(
                go.Scatter(
                    x=cluster_pca[:, 0],
                    y=cluster_pca[:, 1],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    text=cluster_data['CustomerID'],
                    hovertemplate='<b>Cluster %{text}</b><br>PCA1: %{x}<br>PCA2: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Age vs Income scatter
        for cluster in sorted(self.data[cluster_col].unique()):
            cluster_data = self.data[self.data[cluster_col] == cluster]
            
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['Age'],
                    y=cluster_data['AnnualIncome'],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    showlegend=False,
                    hovertemplate='<b>Cluster %{text}</b><br>Age: %{x}<br>Income: %{y}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Spending Score histogram
        for cluster in sorted(self.data[cluster_col].unique()):
            cluster_data = self.data[self.data[cluster_col] == cluster]
            
            fig.add_trace(
                go.Histogram(
                    x=cluster_data['SpendingScore'],
                    name=f'Cluster {cluster}',
                    showlegend=False,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Cluster size bar chart
        cluster_counts = self.data[cluster_col].value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(
                x=[f'Cluster {i}' for i in cluster_counts.index],
                y=cluster_counts.values,
                name='Cluster Size',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=f"Customer Segmentation Analysis - {algorithm.upper()}",
            showlegend=True
        )
        
        return fig
    
    def export_results(self, algorithm='kmeans', filepath=None):
        """Export segmentation results to CSV."""
        if algorithm not in self.results:
            raise ValueError(f"No results found for {algorithm}. Run perform_clustering() first.")
        
        if filepath is None:
            filepath = f'customer_segments_{algorithm}.csv'
        
        # Export data with cluster assignments
        export_data = self.data.copy()
        self.data.to_csv(filepath, index=False)
        
        print(f"Results exported to {filepath}")
        return filepath
    
    def get_segment_recommendations(self, algorithm='kmeans'):
        """Generate marketing recommendations for each segment."""
        if algorithm not in self.results:
            raise ValueError(f"No results found for {algorithm}. Run perform_clustering() first.")
        
        cluster_col = f'{algorithm}_cluster'
        recommendations = {}
        
        for cluster in sorted(self.data[cluster_col].unique()):
            cluster_data = self.data[self.data[cluster_col] == cluster]
            
            avg_age = cluster_data['Age'].mean()
            avg_income = cluster_data['AnnualIncome'].mean()
            avg_spending = cluster_data['SpendingScore'].mean()
            avg_frequency = cluster_data['PurchaseFrequency'].mean()
            
            # Generate recommendations based on characteristics
            if avg_income > 80000 and avg_spending > 70:
                segment_type = "Premium Customers"
                recommendation = "Focus on luxury products, VIP services, and exclusive offers"
            elif avg_age < 35 and avg_frequency > 20:
                segment_type = "Young Frequent Buyers"
                recommendation = "Target with trendy products, social media campaigns, and loyalty programs"
            elif avg_income < 50000 and avg_spending < 40:
                segment_type = "Budget-Conscious"
                recommendation = "Offer discounts, value packages, and budget-friendly options"
            elif avg_age > 50 and avg_frequency < 15:
                segment_type = "Mature Occasional Buyers"
                recommendation = "Focus on quality, reliability, and personalized service"
            else:
                segment_type = "Balanced Customers"
                recommendation = "Maintain engagement with balanced marketing mix"
            
            recommendations[cluster] = {
                'segment_type': segment_type,
                'recommendation': recommendation,
                'size': len(cluster_data),
                'avg_characteristics': {
                    'age': round(avg_age, 1),
                    'income': round(avg_income, 0),
                    'spending_score': round(avg_spending, 1),
                    'purchase_frequency': round(avg_frequency, 1)
                }
            }
        
        return recommendations
