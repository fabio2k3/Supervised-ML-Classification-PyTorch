# 🏠 House Prices Prediction - Advanced Regression Techniques

A complete end-to-end Machine Learning project for the Kaggle House Prices competition using ensemble methods and stacking.

## 📖 Project Description

This project predicts house sale prices using advanced regression techniques and ensemble learning methods. We implement a complete pipeline from exploratory data analysis to model deployment, achieving competitive results through feature engineering and model stacking.

**Competition**: [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**Objective**: Predict the `SalePrice` of residential homes in Ames, Iowa based on 79 explanatory features.

**Final Result**: RMSE of **0.1160** using Stacking Ensemble

---

## 🎯 Project Workflow

### 1. **Exploratory Data Analysis (EDA)**
- Dataset: 1460 training samples, 1459 test samples
- Target variable analysis (SalePrice) - right-skewed distribution
- Missing values analysis (19 features with missing data)
- Feature correlation analysis
- Outlier detection and removal

**Key Findings**:
- Strong correlations: OverallQual (0.79), GrLivArea (0.71), GarageCars (0.64)
- Target requires log transformation to normalize distribution
- 2 outlier properties removed (large area but low price)

### 2. **Data Preprocessing & Feature Engineering**

**Missing Values Handling**:
- Categorical features: Imputed with "None" where NA means "not present"
- Numerical features: Imputed with median or 0 (context-dependent)
- LotFrontage: Imputed with neighborhood median

**Feature Engineering** (11 new features created):
| Feature | Description |
|---------|-------------|
| `TotalSF` | Total square footage (basement + 1st + 2nd floor) |
| `TotalBath` | Total bathrooms (full + 0.5 × half) |
| `HouseAge` | Age of house (YrSold - YearBuilt) |
| `RemodAge` | Years since remodel |
| `WasRemodeled` | Binary indicator of remodeling |
| `TotalQual` | Overall quality + condition |
| `TotalPorchSF` | Total porch area |
| `HasPool`, `HasGarage`, `HasBsmt`, `Has2ndFloor` | Binary indicators |

**Encoding**:
- Ordinal encoding for quality features (Poor → Excellent: 1-5)
- One-hot encoding for nominal categorical features
- Final dataset: **231 features** after preprocessing

**Target Transformation**:
```python
y_train_log = np.log1p(SalePrice)  # Log transformation
```
- Reduced skewness from 1.88 to 0.12
- Improved model performance

### 3. **Model Development**

#### **Base Models** (Individual Benchmarking)

| Model | Type | RMSE (CV) | Description |
|-------|------|-----------|-------------|
| Lasso | Linear L1 | 0.1163 | Regularization with feature selection |
| Ridge | Linear L2 | 0.1161 | Best base model |
| KNN | Instance-based | 0.2152 | k=5, distance-weighted |
| Decision Tree | Tree-based | 0.1794 | max_depth=10 |

#### **Ensemble Models**

| Model | Method | RMSE (CV) | Parameters |
|-------|--------|-----------|------------|
| Random Forest | Bagging | 0.1348 | n_estimators=100, max_depth=20 |
| XGBoost | Boosting | 0.1247 | learning_rate=0.05, n_estimators=100 |
| LightGBM | Boosting | 0.1247 | learning_rate=0.05, n_estimators=100 |

#### **Stacking Ensemble** ⭐

**Architecture**:
```
Base Models (Level 0):
  ├── Lasso (43.09% weight)
  ├── RandomForest (14.88% weight)
  ├── XGBoost (22.16% weight)
  └── LightGBM (21.09% weight)
         ↓
  Meta-Learner (Level 1):
  └── Ridge Regression
```

**Training Process**:
1. Base models generate out-of-fold predictions (5-fold CV)
2. Meta-learner learns optimal combination weights
3. Base models retrained on full training set
4. Meta-learner combines predictions for final output

**Result**: **RMSE = 0.1160** 🏆 (Best overall)

### 4. **Model Evaluation**

**Cross-Validation**: 5-fold stratified

**Metrics**:
- RMSE (Root Mean Squared Error) - primary metric
- MAE (Mean Absolute Error) - secondary metric
- R² Score - goodness of fit

**Final Rankings**:
1. **Stacking** - 0.1160 🥇
2. Ridge - 0.1161
3. Lasso - 0.1163
4. XGBoost - 0.1247
5. LightGBM - 0.1247
6. Random Forest - 0.1348

---

## 🛠️ Technical Implementation

### **Tech Stack**
- **Python 3.8+**
- **scikit-learn** - Base models, preprocessing, CV
- **XGBoost** - Gradient boosting
- **LightGBM** - Gradient boosting
- **pandas, numpy** - Data manipulation
- **matplotlib, seaborn** - Visualization
- **Jupyter** - Interactive analysis

### **Project Structure**
```
house_prices_project/
├── data/
│   ├── train.csv                 # Training data
│   └── test.csv                  # Test data
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory analysis
│   └── 02_feature_engineering.ipynb  # Feature creation
├── src/
│   ├── preprocessing.py          # Data preprocessing pipeline
│   ├── base_models.py           # Base model implementations
│   ├── ensemble_models.py       # Ensemble methods
│   ├── stacking.py              # Stacking ensemble
│   └── evaluation.py            # Metrics and visualization
├── outputs/
│   ├── plots/                   # Generated visualizations
│   ├── submissions/             # Kaggle submission files
│   └── *_results.csv            # Model comparison results
├── train.py                      # Main training script
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## 🚀 How to Run

### **Prerequisites**
1. Download datasets from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
2. Place `train.csv` and `test.csv` in `data/` folder

### **Installation**
```bash
pip install -r requirements.txt
```

### **Execution**

**Complete Pipeline** (recommended):
```bash
python train.py --stage all
```

**Individual Stages**:
```bash
python train.py --stage base_models   # Train base models only
python train.py --stage ensemble      # Train ensemble models only
python train.py --stage stacking      # Train stacking model only
```

**Options**:
```bash
--cv 10          # Change number of CV folds (default: 5)
--no-viz         # Skip visualization generation
```

### **Interactive Analysis**
```bash
jupyter notebook
# Open notebooks/01_eda.ipynb
```

---

## 📊 Results

### **Performance Metrics**

| Metric | Value |
|--------|-------|
| Training RMSE (CV) | 0.1160 |
| Training MAE | 0.0796 |
| Prediction Range | $54K - $962K |
| Mean Prediction | $178K |

### **Feature Importance** (Top 5)
1. OverallQual - Overall material and finish quality
2. GrLivArea - Above grade living area (sq ft)
3. TotalSF - Total square footage (engineered)
4. GarageCars - Size of garage in car capacity
5. TotalBath - Total bathrooms (engineered)

### **Generated Outputs**

After running the pipeline:
```
outputs/
├── training_summary.txt           # Complete training report
├── final_comparison.csv           # All models comparison
├── plots/
│   ├── stacking_predictions_vs_actual.png
│   ├── stacking_residuals.png
│   └── feature_importance.png
└── submissions/
    └── stacking_YYYYMMDD_HHMMSS.csv  # Ready for Kaggle
```

---

## 🎓 Key Learnings

1. **Feature Engineering Impact**: Creating domain-specific features (TotalSF, TotalBath) improved model performance
2. **Log Transformation**: Normalizing skewed target variable was crucial for linear models
3. **Ensemble Methods**: Stacking outperformed individual models by combining their strengths
4. **Missing Data Strategy**: Context-aware imputation (NA as "None" vs. median) preserved data integrity
5. **Model Diversity**: Combining different model types (linear + tree-based) in stacking maximized performance

---

## 📈 Future Improvements

- [ ] Hyperparameter optimization (Grid Search / Bayesian Optimization)
- [ ] Additional feature engineering (polynomial features, interactions)
- [ ] More base models in stacking ensemble
- [ ] Feature selection techniques (RFE, feature importance thresholds)
- [ ] Outlier detection with isolation forests
- [ ] Cross-validation strategy tuning

---

## 📝 Project Timeline

- **Week 1-2**: EDA and data preprocessing
- **Week 3**: Base model development and benchmarking
- **Week 4-5**: Ensemble methods implementation
- **Week 6**: Stacking optimization and final evaluation

