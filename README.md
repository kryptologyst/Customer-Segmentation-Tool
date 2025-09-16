# Customer Segmentation Tool

A comprehensive web-based customer segmentation tool that uses machine learning algorithms to group customers based on their characteristics and behavior patterns. This tool helps businesses understand their customer base and develop targeted marketing strategies.

## Features

- **Interactive Web Interface**: Modern, responsive web UI built with Flask and Bootstrap
- **Multiple Clustering Algorithms**: K-Means, DBSCAN, and Hierarchical clustering
- **Realistic Mock Database**: Generates synthetic customer data with realistic segments
- **Optimal Cluster Detection**: Automatic detection of optimal number of clusters using silhouette analysis
- **Comprehensive Visualizations**: Interactive plots using Plotly for data exploration
- **Segment Analysis**: Detailed statistical analysis of each customer segment
- **Marketing Recommendations**: AI-generated marketing strategies for each segment
- **Data Export**: Export segmentation results to CSV format

## Project Structure

```
0061_Customer_segmentation_tool/
├── app.py                      # Flask web application
├── customer_segmentation.py    # Core segmentation algorithms
├── 0061.py                    # Original basic implementation
├── data/
│   └── mock_database.py       # Mock database generator
├── templates/
│   └── index.html            # Web interface template
├── static/
│   └── js/
│       └── main.js           # Frontend JavaScript
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md               # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 0061_Customer_segmentation_tool
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Follow the step-by-step process**:
   - **Step 1**: Load customer data (specify number of customers)
   - **Step 2**: Select features for clustering
   - **Step 3**: Find optimal number of clusters
   - **Step 4**: Perform segmentation with chosen algorithm
   - **Step 5**: Export results

### Command Line Usage

You can also use the segmentation tool programmatically:

```python
from data.mock_database import CustomerDatabase
from customer_segmentation import CustomerSegmentation

# Generate customer data
db = CustomerDatabase(num_customers=1000)
data = db.load_data()

# Initialize segmentation tool
segmentation = CustomerSegmentation()
segmentation.load_data(dataframe=data)

# Preprocess data
segmentation.preprocess_data()

# Find optimal clusters
optimal_results = segmentation.find_optimal_clusters()
print(f"Optimal clusters: {optimal_results['optimal_k']}")

# Perform clustering
labels = segmentation.perform_clustering(algorithm='kmeans', n_clusters=4)

# Analyze segments
analysis = segmentation.analyze_segments()
recommendations = segmentation.get_segment_recommendations()

# Export results
segmentation.export_results()
```

## Algorithms Supported

### K-Means Clustering
- **Best for**: Spherical clusters, known number of clusters
- **Parameters**: Number of clusters (k)
- **Use case**: General customer segmentation

### DBSCAN
- **Best for**: Irregular shaped clusters, noise detection
- **Parameters**: Epsilon (neighborhood distance), minimum samples
- **Use case**: Identifying outliers and irregular customer groups

### Hierarchical Clustering
- **Best for**: Understanding cluster hierarchy
- **Parameters**: Number of clusters, linkage method
- **Use case**: Detailed customer taxonomy

## Customer Data Features

The mock database generates realistic customer data with the following features:

- **Demographics**: Age, Gender, Location
- **Financial**: Annual Income, Total Spent, Average Order Value
- **Behavioral**: Spending Score, Purchase Frequency, Membership Years
- **Temporal**: Last Purchase Date

## Evaluation Metrics

- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster dispersion (higher is better)
- **Elbow Method**: Identifies optimal number of clusters by analyzing inertia

## Marketing Recommendations

The tool automatically generates marketing recommendations for each segment:

- **Premium Customers**: High income, high spending → Luxury products, VIP services
- **Young Frequent Buyers**: Young, frequent purchases → Trendy products, social media campaigns
- **Budget-Conscious**: Low income, low spending → Discounts, value packages
- **Mature Occasional Buyers**: Older, infrequent purchases → Quality focus, personalized service

## API Endpoints

- `POST /api/load_data`: Load customer data
- `POST /api/preprocess`: Preprocess data for clustering
- `POST /api/find_optimal_clusters`: Find optimal number of clusters
- `POST /api/perform_clustering`: Perform customer segmentation
- `POST /api/export_results`: Export segmentation results
- `GET /api/download_results/<algorithm>`: Download results as CSV

## Dependencies

- **Flask**: Web framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive visualizations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Enhancements

- [ ] Real database integration (PostgreSQL, MongoDB)
- [ ] Advanced clustering algorithms (Gaussian Mixture Models, Spectral Clustering)
- [ ] A/B testing framework for marketing strategies
- [ ] Customer lifetime value prediction
- [ ] Real-time segmentation updates
- [ ] Integration with CRM systems
- [ ] Advanced visualization dashboards
- [ ] Machine learning model deployment

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in `app.py` or kill the process using port 5000
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Data not loading**: Check if the `data` directory exists and has write permissions
4. **Visualization not showing**: Ensure JavaScript is enabled in your browser

### Performance Tips

- For large datasets (>10,000 customers), consider using sampling for initial exploration
- Use DBSCAN for datasets with noise and outliers
- Increase `max_clusters` parameter gradually to avoid long computation times

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the development team.


# Customer-Segmentation-Tool
