"""
Project 61: Customer Segmentation Tool (Basic Implementation)

This is the original basic implementation of the customer segmentation tool.
For the full-featured web application, run: python app.py

Description:
A customer segmentation tool groups customers based on shared characteristics 
such as behavior, demographics, or purchase history. This enables targeted 
marketing and personalized experiences.

This basic version uses K-Means clustering on synthetic customer data to 
identify distinct customer segments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def basic_customer_segmentation():
    """Basic customer segmentation using K-Means clustering."""
    
    print("🎯 Customer Segmentation Tool - Basic Implementation")
    print("=" * 50)
    
    # Generate simulated customer data
    np.random.seed(42)
    data = {
        'Age': np.random.randint(20, 65, 100),
        'AnnualIncome': np.random.randint(20000, 120000, 100),
        'SpendingScore': np.random.randint(1, 100, 100)  # loyalty score or purchase frequency
    }
    df = pd.DataFrame(data)
    
    print(f"📊 Generated {len(df)} customers with {len(df.columns)} features")
    print("\nSample data:")
    print(df.head())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    print("\n✅ Data standardized for clustering")
    
    # Perform K-Means clustering
    k = 4  # Number of customer segments
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Segment'] = model.fit_predict(X_scaled)
    
    print(f"\n🎯 K-Means clustering completed with {k} segments")
    
    # Analyze segments
    segment_analysis = df.groupby('Segment').agg({
        'Age': ['mean', 'std'],
        'AnnualIncome': ['mean', 'std'],
        'SpendingScore': ['mean', 'std']
    }).round(2)
    
    print("\n📈 Segment Analysis:")
    print(segment_analysis)
    
    # Segment sizes
    segment_counts = df['Segment'].value_counts().sort_index()
    print(f"\n👥 Segment Sizes:")
    for segment, count in segment_counts.items():
        print(f"  Segment {segment}: {count} customers ({count/len(df)*100:.1f}%)")
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(X_scaled)
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # PCA plot
    plt.subplot(1, 2, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for cluster in range(k):
        mask = df['Segment'] == cluster
        plt.scatter(pca_features[mask, 0], pca_features[mask, 1], 
                   c=colors[cluster], label=f'Segment {cluster}', alpha=0.7)
    
    plt.title('Customer Segmentation (K-Means + PCA)')
    plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Age vs Income plot
    plt.subplot(1, 2, 2)
    for cluster in range(k):
        mask = df['Segment'] == cluster
        plt.scatter(df[mask]['Age'], df[mask]['AnnualIncome'], 
                   c=colors[cluster], label=f'Segment {cluster}', alpha=0.7)
    
    plt.title('Age vs Annual Income by Segment')
    plt.xlabel('Age')
    plt.ylabel('Annual Income ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n🎉 Basic segmentation complete!")
    print("\n💡 For advanced features, run the web application:")
    print("   python app.py")
    
    return df, model, scaler

if __name__ == "__main__":
    # Run basic segmentation
    customer_data, kmeans_model, data_scaler = basic_customer_segmentation()
    
    print("\n" + "=" * 50)
    print("🧠 What This Project Demonstrates:")
    print("✓ Groups customers into segments using K-Means clustering")
    print("✓ Preprocesses data with standard scaling")
    print("✓ Visualizes high-dimensional clusters with PCA")
    print("✓ Provides segment analysis and statistics")
    print("\n🚀 Enhanced features available in the web app:")
    print("  • Multiple clustering algorithms (DBSCAN, Hierarchical)")
    print("  • Interactive visualizations")
    print("  • Optimal cluster detection")
    print("  • Marketing recommendations")
    print("  • Data export functionality")