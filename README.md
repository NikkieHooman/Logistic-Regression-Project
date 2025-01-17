# Logistic Regression Optimization Project

## Overview
This project explores logistic regression optimization by implementing custom models, comparing them with scikit-learn, and experimenting with advanced techniques like mean square error-based optimization. The goal is to understand the nuances of optimization methods, regularization, and performance tuning in logistic regression.

## Project Structure

### Notebooks

1. **[01_Data_Preparation.ipynb](01_Data_Preparation.ipynb)**
   - Data loading and cleaning.
   - Outlier detection and removal.
   - Multi-class label creation for classification.
   - Feature selection and preprocessing.
   - Dataset splitting into training and testing sets.

2. **[02_Custom_Logistic_Regression.ipynb](02_Custom_Logistic_Regression.ipynb)**
   - Custom logistic regression implementation using numpy.
   - One-vs-all logistic regression for multi-class classification.
   - Regularization options: L1, L2, ElasticNet.
   - Optimization techniques: Steepest Ascent, Stochastic Gradient Ascent, Newton's Method.
   - Parameter tuning and performance evaluation.

3. **[03_Scikit_Learn_Comparison.ipynb](03_Scikit_Learn_Comparison.ipynb)**
   - Logistic regression using scikit-learn.
   - Performance and runtime comparison with custom implementations.

4. **[04_Advanced_Optimization.ipynb](04_Advanced_Optimization.ipynb)**
   - Implementation of logistic regression optimization using mean square error (MSE).
   - Derivation of Hessian matrix for Newton's method.
   - Comparison of MSE-based optimization with traditional methods.

5. **[05_Visualization_and_Results.ipynb](05_Visualization_and_Results.ipynb)**
   - Visualization of tuning results and performance metrics.
   - Final comparisons and deployment recommendations.

## Installation

### Prerequisites
- Python 3.8 or higher
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, scipy, folium

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/username/logistic-regression-project.git
   cd logistic-regression-project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks using Jupyter Notebook or JupyterLab.

## Usage

### Workflow
1. **Data Preparation**:
   - Use `01_Data_Preparation.ipynb` to clean and preprocess the dataset.
2. **Custom Logistic Regression**:
   - Run `02_Custom_Logistic_Regression.ipynb` to test and analyze custom optimization techniques.
3. **Comparison with Scikit-Learn**:
   - Execute `03_Scikit_Learn_Comparison.ipynb` for performance benchmarking.
4. **Advanced Optimization**:
   - Explore MSE-based optimization in `04_Advanced_Optimization.ipynb`.
5. **Results and Visualizations**:
   - Use `05_Visualization_and_Results.ipynb` to interpret findings and derive conclusions.

## Key Findings
- Custom logistic regression achieved ~53% accuracy using Newton's Method with L1 regularization.
- Scikit-learn logistic regression provided ~51% accuracy with significantly faster runtimes.
- MSE-based optimization was less accurate (~40%) but demonstrated the flexibility of custom implementations.

## Deployment Recommendations
- **Production**: Use scikit-learn's implementation for its balance of speed, accuracy, and ease of use.
- **Research**: Custom implementations are ideal for experimenting with advanced optimization techniques and learning.

## Dataset
- Dataset: [Airbnb Prices in European Cities](https://www.kaggle.com/datasets/thedevastator/airbnb-prices-in-european-cities)

## Acknowledgments
- Inspired by coursework and research in machine learning.
- Special thanks to contributors for their effort and insights.
