"""
Flask web application for Customer Segmentation Tool
Provides an interactive web interface for customer segmentation analysis.
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import os
from data.mock_database import CustomerDatabase
from customer_segmentation import CustomerSegmentation
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variables
db = CustomerDatabase()
segmentation_tool = None
current_data = None

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/load_data', methods=['POST'])
def load_data():
    """Load customer data from database."""
    global current_data, segmentation_tool
    
    try:
        # Get parameters from request
        data = request.get_json()
        num_customers = data.get('num_customers', 1000)
        
        # Load or generate data
        db.num_customers = num_customers
        current_data = db.load_data()
        
        # Initialize segmentation tool
        segmentation_tool = CustomerSegmentation()
        segmentation_tool.load_data(dataframe=current_data)
        
        # Get basic statistics
        stats = {
            'total_customers': len(current_data),
            'features': list(current_data.columns),
            'numeric_features': current_data.select_dtypes(include=[np.number]).columns.tolist(),
            'sample_data': current_data.head(10).to_dict('records')
        }
        
        return jsonify({
            'success': True,
            'message': f'Loaded {len(current_data)} customers successfully',
            'data': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error loading data: {str(e)}'
        }), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """Preprocess data for clustering."""
    global segmentation_tool
    
    if segmentation_tool is None:
        return jsonify({
            'success': False,
            'message': 'No data loaded. Please load data first.'
        }), 400
    
    try:
        data = request.get_json()
        selected_features = data.get('features', None)
        
        # Preprocess data
        segmentation_tool.preprocess_data(features=selected_features)
        
        return jsonify({
            'success': True,
            'message': 'Data preprocessed successfully',
            'features_used': segmentation_tool.feature_columns
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error preprocessing data: {str(e)}'
        }), 500

@app.route('/api/find_optimal_clusters', methods=['POST'])
def find_optimal_clusters():
    """Find optimal number of clusters."""
    global segmentation_tool
    
    if segmentation_tool is None or segmentation_tool.scaled_data is None:
        return jsonify({
            'success': False,
            'message': 'Data not preprocessed. Please preprocess data first.'
        }), 400
    
    try:
        data = request.get_json()
        max_clusters = data.get('max_clusters', 10)
        method = data.get('method', 'kmeans')
        
        # Find optimal clusters
        results = segmentation_tool.find_optimal_clusters(max_clusters=max_clusters, method=method)
        
        # Create elbow plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results['k_range'],
            y=results['inertias'],
            mode='lines+markers',
            name='Inertia',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=results['k_range'],
            y=results['silhouette_scores'],
            mode='lines+markers',
            name='Silhouette Score',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Optimal Number of Clusters Analysis',
            xaxis_title='Number of Clusters',
            yaxis=dict(title='Inertia', side='left'),
            yaxis2=dict(title='Silhouette Score', side='right', overlaying='y'),
            hovermode='x unified'
        )
        
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True,
            'message': f'Optimal number of clusters: {results["optimal_k"]}',
            'results': results,
            'plot': plot_json
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error finding optimal clusters: {str(e)}'
        }), 500

@app.route('/api/perform_clustering', methods=['POST'])
def perform_clustering():
    """Perform customer segmentation."""
    global segmentation_tool
    
    if segmentation_tool is None or segmentation_tool.scaled_data is None:
        return jsonify({
            'success': False,
            'message': 'Data not preprocessed. Please preprocess data first.'
        }), 400
    
    try:
        data = request.get_json()
        algorithm = data.get('algorithm', 'kmeans')
        n_clusters = data.get('n_clusters', 4)
        
        # Additional parameters for different algorithms
        kwargs = {}
        if algorithm == 'dbscan':
            kwargs['eps'] = data.get('eps', 0.5)
            kwargs['min_samples'] = data.get('min_samples', 5)
        
        # Perform clustering
        labels = segmentation_tool.perform_clustering(
            algorithm=algorithm, 
            n_clusters=n_clusters, 
            **kwargs
        )
        
        # Get analysis results
        segment_analysis = segmentation_tool.analyze_segments(algorithm=algorithm)
        recommendations = segmentation_tool.get_segment_recommendations(algorithm=algorithm)
        
        # Create visualizations
        fig = segmentation_tool.create_visualizations(algorithm=algorithm)
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'success': True,
            'message': f'Clustering completed with {algorithm}',
            'results': segmentation_tool.results[algorithm],
            'segment_analysis': segment_analysis.to_dict(),
            'recommendations': recommendations,
            'plot': plot_json
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error performing clustering: {str(e)}'
        }), 500

@app.route('/api/export_results', methods=['POST'])
def export_results():
    """Export segmentation results."""
    global segmentation_tool
    
    if segmentation_tool is None:
        return jsonify({
            'success': False,
            'message': 'No segmentation results available.'
        }), 400
    
    try:
        data = request.get_json()
        algorithm = data.get('algorithm', 'kmeans')
        
        # Export results
        filepath = segmentation_tool.export_results(algorithm=algorithm)
        
        return jsonify({
            'success': True,
            'message': f'Results exported to {filepath}',
            'filepath': filepath
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error exporting results: {str(e)}'
        }), 500

@app.route('/api/download_results/<algorithm>')
def download_results(algorithm):
    """Download segmentation results as CSV."""
    global segmentation_tool
    
    if segmentation_tool is None:
        return jsonify({'error': 'No data available'}), 404
    
    try:
        # Create CSV content
        output = io.StringIO()
        segmentation_tool.data.to_csv(output, index=False)
        output.seek(0)
        
        # Create file-like object
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            as_attachment=True,
            download_name=f'customer_segments_{algorithm}.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data_overview')
def data_overview():
    """Get overview of current data."""
    global current_data
    
    if current_data is None:
        return jsonify({
            'success': False,
            'message': 'No data loaded.'
        }), 400
    
    try:
        # Basic statistics
        overview = {
            'shape': current_data.shape,
            'columns': list(current_data.columns),
            'dtypes': current_data.dtypes.astype(str).to_dict(),
            'missing_values': current_data.isnull().sum().to_dict(),
            'summary_stats': current_data.describe().to_dict()
        }
        
        return jsonify({
            'success': True,
            'overview': overview
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting data overview: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
