"""
Main training script for House Prices Competition
Runs the full pipeline: preprocessing, base models, ensemble and stacking

Usage:
    python train.py --stage all
    python train.py --stage base_models
    python train.py --stage ensemble
    python train.py --stage stacking
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from preprocessing import HousePricePreprocessor
from base_models import BaseModels
from ensemble_models import EnsembleModels
from stacking import StackingRegressor
from evaluation import ModelEvaluator, create_kaggle_submission


def setup_directories():
    """Create required output directories"""
    directories = ['outputs', 'outputs/models', 'outputs/plots', 'outputs/submissions']
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Directories configured")


def load_and_preprocess_data(train_path='data/train.csv', test_path='data/test.csv'):
    """
    Load and preprocess data
    
    Returns:
        X_train, y_train, X_test, test_ids, preprocessor
    """
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    preprocessor = HousePricePreprocessor()
    
    # Load data
    train_df, test_df = preprocessor.load_data(train_path, test_path)
    
    # Apply full preprocessing pipeline
    X_train, y_train, X_test, test_ids = preprocessor.preprocess_pipeline(
        train_df, test_df, target_col='SalePrice'
    )
    
    # Convert to numpy arrays if needed
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        X_train = X_train.values
    else:
        feature_names = preprocessor.feature_names
    
    if X_test is not None and isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    
    print(f"\n✅ Preprocessing completed")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    if X_test is not None:
        print(f"   X_test shape: {X_test.shape}")
    print(f"   Number of features: {len(feature_names)}")
    
    return X_train, y_train, X_test, test_ids, preprocessor, feature_names


def train_base_models(X_train, y_train, cv=5):
    """
    Train and evaluate base models
    """
    print("\n" + "="*70)
    print("STEP 2: BASE MODELS TRAINING")
    print("="*70)
    
    base_models = BaseModels(random_state=42)
    base_models.initialize_models()
    
    results = base_models.train_and_evaluate_all(X_train, y_train, cv=cv)
    
    results_table = base_models.get_results_table()
    print("\n📊 BASE MODELS RESULTS:")
    print(results_table.to_string(index=False))
    
    results_table.to_csv('outputs/base_models_results.csv', index=False)
    print("\n✅ Results saved to outputs/base_models_results.csv")
    
    best_name, best_model = base_models.get_best_model()
    print(f"\n🏆 Best base model: {best_name}")
    print(f"   RMSE: {results[best_name]['RMSE_mean']:.4f}")
    
    return base_models, results


def train_ensemble_models(X_train, y_train, cv=5):
    """
    Train and evaluate ensemble models
    """
    print("\n" + "="*70)
    print("STEP 3: ENSEMBLE MODELS TRAINING")
    print("="*70)
    
    ensemble_models = EnsembleModels(random_state=42)
    ensemble_models.initialize_models()
    
    results = ensemble_models.train_and_evaluate_all(X_train, y_train, cv=cv)
    
    results_table = ensemble_models.get_results_table()
    print("\n📊 ENSEMBLE MODELS RESULTS:")
    print(results_table.to_string(index=False))
    
    results_table.to_csv('outputs/ensemble_models_results.csv', index=False)
    print("\n✅ Results saved to outputs/ensemble_models_results.csv")
    
    best_name, best_model = ensemble_models.get_best_model()
    print(f"\n🏆 Best ensemble model: {best_name}")
    print(f"   RMSE: {results[best_name]['RMSE_mean']:.4f}")
    
    return ensemble_models, results


def train_stacking_model(X_train, y_train, cv=5):
    """
    Train stacking model
    """
    print("\n" + "="*70)
    print("STEP 4: STACKING ENSEMBLE TRAINING")
    print("="*70)
    
    stacking = StackingRegressor(random_state=42, cv=cv)
    stacking.initialize_models()
    stacking.fit(X_train, y_train)
    
    weights = stacking.get_meta_weights()
    if weights:
        print("\n📊 META-LEARNER WEIGHTS (Ridge):")
        for model_name, weight in weights.items():
            print(f"   {model_name}: {weight:.4f}")
    
    return stacking


def compare_all_models(X_train, y_train, base_models, ensemble_models, stacking, cv=5):
    """
    Compare all trained models
    """
    print("\n" + "="*70)
    print("STEP 5: FINAL MODEL COMPARISON")
    print("="*70)
    
    from sklearn.model_selection import cross_val_score
    
    all_results = []
    
    for name, model in base_models.models.items():
        mse_scores = -cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rmse_mean = np.sqrt(mse_scores.mean())
        rmse_std = np.sqrt(mse_scores.std())
        
        all_results.append({
            'Model': name,
            'Type': 'Base',
            'RMSE_mean': rmse_mean,
            'RMSE_std': rmse_std
        })
    
    for name, model in ensemble_models.models.items():
        mse_scores = -cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rmse_mean = np.sqrt(mse_scores.mean())
        rmse_std = np.sqrt(mse_scores.std())
        
        model_type = 'Bagging' if 'Forest' in name else 'Boosting'
        all_results.append({
            'Model': name,
            'Type': model_type,
            'RMSE_mean': rmse_mean,
            'RMSE_std': rmse_std
        })
    
    stacking_pred = stacking.meta_model.predict(stacking.base_predictions_train)
    stacking_rmse = np.sqrt(mean_squared_error(y_train, stacking_pred))
    
    all_results.append({
        'Model': 'Stacking',
        'Type': 'Meta-Ensemble',
        'RMSE_mean': stacking_rmse,
        'RMSE_std': 0.0
    })
    
    comparison_df = pd.DataFrame(all_results).sort_values('RMSE_mean')
    
    print("\n📊 FINAL COMPARISON (sorted by RMSE):")
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv('outputs/final_comparison.csv', index=False)
    print("\n✅ Comparison saved to outputs/final_comparison.csv")
    
    best_model = comparison_df.iloc[0]
    print(f"\n🏆 BEST OVERALL MODEL: {best_model['Model']}")
    print(f"   Type: {best_model['Type']}")
    print(f"   RMSE: {best_model['RMSE_mean']:.4f} (+/- {best_model['RMSE_std']:.4f})")
    
    return comparison_df



def generate_predictions(X_test, test_ids, preprocessor, stacking, output_name='stacking'):
    """
    Generate predictions for Kaggle submission
    """
    print("\n" + "="*70)
    print("STEP 6: GENERATING PREDICTIONS FOR KAGGLE")
    print("="*70)
    
    if X_test is None:
        print("⚠️  No test data available")
        return None
    
    # Predictions in log scale
    predictions_log = stacking.predict(X_test)
    
    # Inverse log transformation
    predictions = preprocessor.inverse_transform_target(predictions_log)
    
    # Create submission file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'outputs/submissions/{output_name}_{timestamp}.csv'
    
    submission = create_kaggle_submission(test_ids, predictions, filename)
    
    print(f"\n✅ Submission created: {filename}")
    print("   Prediction statistics:")
    print(f"   - Min:    ${predictions.min():,.2f}")
    print(f"   - Max:    ${predictions.max():,.2f}")
    print(f"   - Mean:   ${predictions.mean():,.2f}")
    print(f"   - Median: ${np.median(predictions):,.2f}")
    
    return submission


def create_visualizations(X_train, y_train, base_models, ensemble_models, 
                         stacking, feature_names, preprocessor):
    """
    Generate analysis visualizations
    """
    print("\n" + "="*70)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("="*70)
    
    evaluator = ModelEvaluator()
    
    # Stacking predictions for visualization
    from sklearn.model_selection import cross_val_predict, KFold
    
    # Use out-of-fold stacking predictions
    stacking_pred = stacking.meta_model.predict(stacking.base_predictions_train)
    
    # Inverse log transform for visualization
    y_train_original = preprocessor.inverse_transform_target(y_train)
    stacking_pred_original = preprocessor.inverse_transform_target(stacking_pred)
    
    print("\n📊 Generating plots...")
    
    # 1. Predictions vs Actual
    evaluator.plot_predictions_vs_actual(
        y_train_original, 
        stacking_pred_original,
        model_name='Stacking Ensemble',
        save_path='outputs/plots/stacking_predictions_vs_actual.png'
    )
    
    # 2. Residual plots
    evaluator.plot_residuals(
        y_train_original,
        stacking_pred_original,
        model_name='Stacking Ensemble',
        save_path='outputs/plots/stacking_residuals.png'
    )
    
    # 3. Feature importance from best ensemble model
    best_ensemble_name, best_ensemble_model = ensemble_models.get_best_model()
    if hasattr(best_ensemble_model, 'feature_importances_'):
        evaluator.plot_feature_importance(
            best_ensemble_model,
            feature_names,
            top_n=20,
            model_name=best_ensemble_name,
            save_path=f'outputs/plots/{best_ensemble_name.lower()}_feature_importance.png'
        )
    
    print("\n✅ Visualizations saved to outputs/plots/")


def save_training_summary(comparison_df, preprocessor):
    """
    Save training summary
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = f"""
{'='*70}
TRAINING SUMMARY - HOUSE PRICES COMPETITION
{'='*70}

Date: {timestamp}

PREPROCESSING:
- Original features: 79
- Final features: {len(preprocessor.feature_names)}
- Target transformation: log(SalePrice)
- Missing values: Imputed
- Outliers: Removed

TRAINED MODELS:
{comparison_df.to_string(index=False)}

BEST MODEL: {comparison_df.iloc[0]['Model']}
- RMSE: {comparison_df.iloc[0]['RMSE_mean']:.4f}
- Type: {comparison_df.iloc[0]['Type']}

GENERATED FILES:
- outputs/base_models_results.csv
- outputs/ensemble_models_results.csv
- outputs/final_comparison.csv
- outputs/plots/*.png
- outputs/submissions/*.csv

{'='*70}
    """
    
    with open('outputs/training_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\n✅ Training summary saved to outputs/training_summary.txt")
    print(summary)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train House Prices models')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['all', 'base_models', 'ensemble', 'stacking'],
                       help='Training stage to run')
    parser.add_argument('--train', type=str, default='data/train.csv',
                       help='Path to train.csv file')
    parser.add_argument('--test', type=str, default='data/test.csv',
                       help='Path to test.csv file')
    parser.add_argument('--cv', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--no-viz', action='store_true',
                       help='Do not generate visualizations')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HOUSE PRICES - ADVANCED REGRESSION TECHNIQUES")
    print("Full Training Pipeline")
    print("="*70)
    
    # Setup
    setup_directories()
    
    # Load and preprocess
    X_train, y_train, X_test, test_ids, preprocessor, feature_names = load_and_preprocess_data(
        args.train, args.test
    )
    
    base_models_obj = None
    ensemble_models_obj = None
    stacking_obj = None
    
    if args.stage in ['all', 'base_models']:
        base_models_obj, base_results = train_base_models(X_train, y_train, args.cv)
    
    if args.stage in ['all', 'ensemble']:
        ensemble_models_obj, ensemble_results = train_ensemble_models(X_train, y_train, args.cv)
    
    if args.stage in ['all', 'stacking']:
        stacking_obj = train_stacking_model(X_train, y_train, args.cv)
    
    if args.stage == 'all':
        comparison_df = compare_all_models(
            X_train, y_train, 
            base_models_obj, 
            ensemble_models_obj, 
            stacking_obj, 
            args.cv
        )
        
        if X_test is not None:
            generate_predictions(X_test, test_ids, preprocessor, stacking_obj)
        
        if not args.no_viz:
            create_visualizations(
                X_train, y_train,
                base_models_obj,
                ensemble_models_obj,
                stacking_obj,
                feature_names,
                preprocessor
            )
        
        save_training_summary(comparison_df, preprocessor)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nNext steps:")
    print("1. Review outputs/training_summary.txt")
    print("2. Analyze plots in outputs/plots/")
    print("3. Upload submission to Kaggle from outputs/submissions/")
    print("\n")


if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error
    main()
