// Main JavaScript for Customer Segmentation Tool

class CustomerSegmentationApp {
    constructor() {
        this.currentStep = 1;
        this.currentAlgorithm = 'kmeans';
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateStepIndicator();
    }

    bindEvents() {
        // Load Data
        document.getElementById('load-data-btn').addEventListener('click', () => {
            this.loadData();
        });

        // Preprocess Data
        document.getElementById('preprocess-btn').addEventListener('click', () => {
            this.preprocessData();
        });

        // Find Optimal Clusters
        document.getElementById('find-clusters-btn').addEventListener('click', () => {
            this.findOptimalClusters();
        });

        // Perform Clustering
        document.getElementById('perform-clustering-btn').addEventListener('click', () => {
            this.performClustering();
        });

        // Algorithm change
        document.getElementById('algorithm').addEventListener('change', (e) => {
            this.currentAlgorithm = e.target.value;
            this.toggleAlgorithmParams();
        });

        // Export Results
        document.getElementById('export-btn').addEventListener('click', () => {
            this.exportResults();
        });
    }

    showLoading(buttonId) {
        const button = document.getElementById(buttonId);
        const loading = button.querySelector('.loading');
        loading.style.display = 'inline-block';
        button.disabled = true;
    }

    hideLoading(buttonId) {
        const button = document.getElementById(buttonId);
        const loading = button.querySelector('.loading');
        loading.style.display = 'none';
        button.disabled = false;
    }

    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alert-container');
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        alertContainer.appendChild(alertDiv);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    updateStepIndicator() {
        for (let i = 1; i <= 5; i++) {
            const step = document.getElementById(`step-${i}`);
            step.classList.remove('active', 'completed');
            
            if (i < this.currentStep) {
                step.classList.add('completed');
            } else if (i === this.currentStep) {
                step.classList.add('active');
            }
        }
    }

    async loadData() {
        this.showLoading('load-data-btn');
        
        try {
            const numCustomers = document.getElementById('num-customers').value;
            
            const response = await fetch('/api/load_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    num_customers: parseInt(numCustomers)
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showAlert(result.message, 'success');
                this.displayDataOverview(result.data);
                this.currentStep = 2;
                this.updateStepIndicator();
                document.getElementById('preprocess-section').style.display = 'block';
                this.populateFeatureSelection(result.data.numeric_features);
            } else {
                this.showAlert(result.message, 'danger');
            }
        } catch (error) {
            this.showAlert(`Error: ${error.message}`, 'danger');
        } finally {
            this.hideLoading('load-data-btn');
        }
    }

    displayDataOverview(data) {
        const overviewDiv = document.getElementById('data-overview');
        const statsDiv = document.getElementById('data-stats');
        
        statsDiv.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="card bg-light">
                        <div class="card-body text-center">
                            <h5 class="card-title">${data.total_customers}</h5>
                            <p class="card-text">Total Customers</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card bg-light">
                        <div class="card-body text-center">
                            <h5 class="card-title">${data.features.length}</h5>
                            <p class="card-text">Features</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card bg-light">
                        <div class="card-body text-center">
                            <h5 class="card-title">${data.numeric_features.length}</h5>
                            <p class="card-text">Numeric Features</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        overviewDiv.style.display = 'block';
    }

    populateFeatureSelection(numericFeatures) {
        const featureDiv = document.getElementById('feature-selection');
        const defaultFeatures = ['Age', 'AnnualIncome', 'SpendingScore', 'PurchaseFrequency', 'TotalSpent'];
        
        featureDiv.innerHTML = numericFeatures.map(feature => `
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="checkbox" id="feature-${feature}" value="${feature}" 
                       ${defaultFeatures.includes(feature) ? 'checked' : ''}>
                <label class="form-check-label" for="feature-${feature}">
                    ${feature}
                </label>
            </div>
        `).join('');
    }

    async preprocessData() {
        this.showLoading('preprocess-btn');
        
        try {
            const selectedFeatures = Array.from(document.querySelectorAll('#feature-selection input:checked'))
                .map(input => input.value);

            if (selectedFeatures.length === 0) {
                this.showAlert('Please select at least one feature for clustering.', 'warning');
                return;
            }

            const response = await fetch('/api/preprocess', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    features: selectedFeatures
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showAlert(result.message, 'success');
                this.currentStep = 3;
                this.updateStepIndicator();
                document.getElementById('optimal-clusters-section').style.display = 'block';
            } else {
                this.showAlert(result.message, 'danger');
            }
        } catch (error) {
            this.showAlert(`Error: ${error.message}`, 'danger');
        } finally {
            this.hideLoading('preprocess-btn');
        }
    }

    async findOptimalClusters() {
        this.showLoading('find-clusters-btn');
        
        try {
            const maxClusters = document.getElementById('max-clusters').value;
            
            const response = await fetch('/api/find_optimal_clusters', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    max_clusters: parseInt(maxClusters),
                    method: 'kmeans'
                })
            });

            const result = await response.json();

            if (result.success) {
                this.showAlert(result.message, 'success');
                
                // Display plot
                const plotDiv = document.getElementById('optimal-clusters-plot');
                Plotly.newPlot(plotDiv, JSON.parse(result.plot).data, JSON.parse(result.plot).layout);
                
                // Update recommended clusters
                document.getElementById('n-clusters').value = result.results.optimal_k;
                
                // Display results
                const resultDiv = document.getElementById('optimal-clusters-result');
                resultDiv.innerHTML = `
                    <div class="alert alert-info">
                        <strong>Recommended number of clusters:</strong> ${result.results.optimal_k}
                        <br>
                        <small>Based on silhouette score analysis</small>
                    </div>
                `;
                
                this.currentStep = 4;
                this.updateStepIndicator();
                document.getElementById('clustering-section').style.display = 'block';
            } else {
                this.showAlert(result.message, 'danger');
            }
        } catch (error) {
            this.showAlert(`Error: ${error.message}`, 'danger');
        } finally {
            this.hideLoading('find-clusters-btn');
        }
    }

    toggleAlgorithmParams() {
        const dbscanParams = document.getElementById('dbscan-params');
        const nClustersInput = document.getElementById('n-clusters').parentElement;
        
        if (this.currentAlgorithm === 'dbscan') {
            dbscanParams.style.display = 'block';
            nClustersInput.style.display = 'none';
        } else {
            dbscanParams.style.display = 'none';
            nClustersInput.style.display = 'block';
        }
    }

    async performClustering() {
        this.showLoading('perform-clustering-btn');
        
        try {
            const algorithm = document.getElementById('algorithm').value;
            const nClusters = document.getElementById('n-clusters').value;
            
            let requestData = {
                algorithm: algorithm,
                n_clusters: parseInt(nClusters)
            };

            // Add DBSCAN specific parameters
            if (algorithm === 'dbscan') {
                requestData.eps = parseFloat(document.getElementById('eps').value);
                requestData.min_samples = parseInt(document.getElementById('min-samples').value);
            }

            const response = await fetch('/api/perform_clustering', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            if (result.success) {
                this.showAlert(result.message, 'success');
                this.displayClusteringResults(result);
                this.currentStep = 5;
                this.updateStepIndicator();
                document.getElementById('export-section').style.display = 'block';
            } else {
                this.showAlert(result.message, 'danger');
            }
        } catch (error) {
            this.showAlert(`Error: ${error.message}`, 'danger');
        } finally {
            this.hideLoading('perform-clustering-btn');
        }
    }

    displayClusteringResults(result) {
        // Show results section
        document.getElementById('clustering-results').style.display = 'block';
        
        // Display metrics
        const metricsDiv = document.getElementById('clustering-metrics');
        metricsDiv.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="card bg-light">
                        <div class="card-body text-center">
                            <h5 class="card-title">${result.results.n_clusters}</h5>
                            <p class="card-text">Clusters Found</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card bg-light">
                        <div class="card-body text-center">
                            <h5 class="card-title">${result.results.silhouette_score.toFixed(3)}</h5>
                            <p class="card-text">Silhouette Score</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card bg-light">
                        <div class="card-body text-center">
                            <h5 class="card-title">${result.results.calinski_harabasz_score.toFixed(1)}</h5>
                            <p class="card-text">Calinski-Harabasz Score</p>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Display plot
        const plotDiv = document.getElementById('clustering-plot');
        Plotly.newPlot(plotDiv, JSON.parse(result.plot).data, JSON.parse(result.plot).layout);

        // Display segment analysis
        this.displaySegmentAnalysis(result.segment_analysis);
        
        // Display recommendations
        this.displayRecommendations(result.recommendations);
    }

    displaySegmentAnalysis(segmentAnalysis) {
        const analysisDiv = document.getElementById('segment-analysis');
        
        // Convert to table format
        const clusters = Object.keys(segmentAnalysis);
        const metrics = Object.keys(segmentAnalysis[clusters[0]]);
        
        let tableHTML = `
            <div class="table-responsive">
                <table class="table table-striped table-hover">
                    <thead class="table-dark">
                        <tr>
                            <th>Cluster</th>
                            ${metrics.map(metric => `<th>${metric.replace(/_/g, ' ')}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        clusters.forEach(cluster => {
            tableHTML += `
                <tr>
                    <td><strong>Cluster ${cluster}</strong></td>
                    ${metrics.map(metric => `<td>${segmentAnalysis[cluster][metric]}</td>`).join('')}
                </tr>
            `;
        });
        
        tableHTML += `
                    </tbody>
                </table>
            </div>
        `;
        
        analysisDiv.innerHTML = tableHTML;
    }

    displayRecommendations(recommendations) {
        const recommendationsDiv = document.getElementById('recommendations');
        
        let cardsHTML = '<div class="row">';
        
        Object.keys(recommendations).forEach(cluster => {
            const rec = recommendations[cluster];
            cardsHTML += `
                <div class="col-md-6 mb-3">
                    <div class="card h-100">
                        <div class="card-header bg-primary text-white">
                            <h6 class="mb-0">Cluster ${cluster}: ${rec.segment_type}</h6>
                        </div>
                        <div class="card-body">
                            <p class="card-text">${rec.recommendation}</p>
                            <small class="text-muted">
                                Size: ${rec.size} customers<br>
                                Avg Age: ${rec.avg_characteristics.age}<br>
                                Avg Income: $${rec.avg_characteristics.income.toLocaleString()}<br>
                                Avg Spending Score: ${rec.avg_characteristics.spending_score}
                            </small>
                        </div>
                    </div>
                </div>
            `;
        });
        
        cardsHTML += '</div>';
        recommendationsDiv.innerHTML = cardsHTML;
    }

    async exportResults() {
        try {
            const algorithm = this.currentAlgorithm;
            const response = await fetch(`/api/download_results/${algorithm}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `customer_segments_${algorithm}.csv`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.showAlert('Results exported successfully!', 'success');
            } else {
                this.showAlert('Error exporting results', 'danger');
            }
        } catch (error) {
            this.showAlert(`Error: ${error.message}`, 'danger');
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CustomerSegmentationApp();
});
