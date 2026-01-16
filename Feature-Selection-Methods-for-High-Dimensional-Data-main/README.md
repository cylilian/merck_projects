# Feature Selection Methods for High-Dimensional Data 🔍

A comprehensive demonstration of three popular feature selection techniques applied to synthetic high-dimensional datasets using Python and scikit-learn.

## 📋 Overview

This notebook demonstrates the practical implementation of three key feature selection methods:

1. **Symmetrical Uncertainty (SU) Filter** - Information-theoretic approach
2. **Minimum Redundancy Maximum Relevance (mRMR)** - Balances relevance and redundancy
3. **ElasticNet Embedded Method** - L1/L2 regularization for feature selection

## 🎯 Dataset

- **Samples**: 50 observations
- **Features**: 10 total features
- **Informative**: 5 features contain useful signal
- **Redundant**: 2 features are redundant
- **Target**: Binary classification problem

## 📁 Project Structure

```
feature-selection-demo/
├── featireselection.ipynb          # Main notebook with all implementations
├── README.md                       # This documentation
└── requirements.txt                # Project dependencies
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/feature-selection-demo.git
   cd feature-selection-demo
   ```

2. **Install required packages:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook featireselection.ipynb
   ```

## 🔧 Methods Implemented

### 1. Symmetrical Uncertainty (SU) Filter
```python
# Approximates SU using normalized mutual information
su_scores = 2 * mi_scores / (np.log2(var_X) + np.log2(var_y))
```
- **Purpose**: Measures information gain between features and target
- **Selection**: Top 10% features (90th percentile threshold)
- **Advantage**: Fast, model-agnostic

### 2. Minimum Redundancy Maximum Relevance (mRMR)
```python
# Custom implementation balancing relevance vs redundancy
def mrmr_select(X, y, k=10):
    # Iteratively selects features with high relevance, low redundancy
```
- **Purpose**: Selects features highly correlated with target but minimally redundant
- **Selection**: Iterative greedy algorithm
- **Advantage**: Considers feature interactions

### 3. ElasticNet Embedded Method
```python
# Combines L1 and L2 penalties for automatic feature selection
enet = ElasticNetCV(l1_ratio=0.9, cv=5, random_state=0)
```
- **Purpose**: Embedded feature selection during model training
- **Selection**: Features with non-zero coefficients
- **Advantage**: Model-specific feature importance

## 📊 Notebook Workflow

The notebook follows a clear 6-step process:

### Step 1: Data Preparation
- Generate synthetic dataset with controlled properties
- Create DataFrame with feature names (F0, F1, F2, ...)
- Display dataset shape and preview

### Step 2: Symmetrical Uncertainty (SU) Analysis
- Calculate mutual information scores
- Compute SU scores using information theory
- Visualize feature importance with bar chart
- Select top features based on 90th percentile threshold

### Step 3: mRMR Implementation
- Custom function for iterative feature selection
- Balance between relevance (MI with target) and redundancy (MI between features)
- Greedy selection of k=10 features

### Step 4: ElasticNet Embedded Selection
- Train ElasticNet with cross-validation
- Extract features with non-zero coefficients
- Visualize coefficient values

### Step 5: Method Comparison
- Create scatter plot comparing all three methods
- Show selected features on same SU score scale
- Identify consensus and unique selections

### Step 6: Correlation Analysis
- Generate heatmap of selected features
- Analyze relationships between chosen features

## 📈 Visualizations

The notebook includes several informative plots:

1. **SU Scores Bar Chart**: Shows feature importance scores across all features
2. **ElasticNet Coefficients**: Bar plot of feature weights from regularization
3. **Method Comparison Scatter**: Compares feature selections across all three methods
4. **Correlation Heatmap**: Shows relationships between SU-selected features

## � Key Code Snippets

### Data Generation
```python
X, y = make_classification(
    n_samples=50, n_features=10, n_informative=5, 
    n_redundant=2, random_state=42
)
```

### SU Score Calculation
```python
mi_scores = mutual_info_classif(X, y, random_state=0)
su_scores = 2 * mi_scores / (np.log2(var_X) + np.log2(var_y))
```

### mRMR Selection
```python
def mrmr_select(X, y, k=10):
    mi = mutual_info_classif(X, y, random_state=0)
    selected = [np.argmax(mi)]
    # Iterative selection balancing relevance vs redundancy
```

## 📚 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## � Educational Value

Perfect for:
- Learning feature selection fundamentals
- Comparing filter vs embedded methods  
- Understanding information theory in ML
- Hands-on experience with scikit-learn
- Feature engineering best practices

## 💡 Key Insights

- **SU Filter**: Fast baseline using information theory
- **mRMR**: Better handles feature redundancy through iterative selection
- **ElasticNet**: Provides model-specific feature importance
- **Method Comparison**: Different approaches select different feature subsets
- **Visualization**: Critical for understanding feature relationships

<<<<<<< HEAD
## 🔍 Use Cases

- **High-dimensional datasets** with many irrelevant features
- **Preprocessing step** before machine learning models
- **Feature engineering** exploration and validation
- **Educational purposes** for understanding selection methods
- **Comparative analysis** of different selection strategies

## 🔗 Further Reading
=======
## 🤝 Contributing
>>>>>>> 70311cfb511cff6e61ac2bf485cb3826a055623b

- [Feature Selection Guide - scikit-learn](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Mutual Information Theory](https://en.wikipedia.org/wiki/Mutual_information)
- [ElasticNet Regularization](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)
- [mRMR Algorithm Details](https://en.wikipedia.org/wiki/Feature_selection#Minimum_redundancy_feature_selection)

---

**⭐ Star this repository if you found it helpful for learning feature selection methods!**
