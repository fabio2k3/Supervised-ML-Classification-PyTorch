"""
Módulo de evaluación de modelos
Métricas: RMSE, MAE, R2
Visualizaciones y análisis de errores
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Clase para evaluación completa de modelos de regresión
    """
    
    def __init__(self):
        self.results = {}
        
    def calculate_metrics(self, y_true, y_pred, model_name='Model'):
        """
        Calcula métricas de regresión
        
        Returns:
            dict: RMSE, MAE, R2
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        self.results[model_name] = metrics
        
        return metrics
    
    def print_metrics(self, y_true, y_pred, model_name='Model'):
        """Imprime métricas de manera formateada"""
        metrics = self.calculate_metrics(y_true, y_pred, model_name)
        
        print(f"\n{'='*50}")
        print(f"MÉTRICAS: {model_name}")
        print(f"{'='*50}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE:  {metrics['MAE']:.4f}")
        print(f"R²:   {metrics['R2']:.4f}")
        print(f"{'='*50}\n")
        
        return metrics
    
    def compare_models(self, predictions_dict, y_true):
        """
        Compara múltiples modelos
        
        Args:
            predictions_dict: {model_name: predictions}
            y_true: valores reales
            
        Returns:
            DataFrame con comparación de métricas
        """
        comparison_data = []
        
        for model_name, y_pred in predictions_dict.items():
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE')
        
        return comparison_df
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name='Model', 
                                   save_path=None):
        """
        Gráfico de predicciones vs valores reales
        """
        plt.figure(figsize=(10, 6))
        
        plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
        
        # Línea de predicción perfecta
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(f'{model_name}: Predictions vs Actual', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
    def plot_residuals(self, y_true, y_pred, model_name='Model', save_path=None):
        """
        Gráfico de residuos
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuos vs predicciones
        axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Values', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title(f'{model_name}: Residual Plot', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Histograma de residuos
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Residuals', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title(f'{model_name}: Residual Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, top_n=20, 
                                model_name='Model', save_path=None):
        """
        Gráfico de importancia de features
        """
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️  {model_name} no tiene atributo feature_importances_")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices], align='center')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'{model_name}: Top {top_n} Feature Importances', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, comparison_df, metric='RMSE', save_path=None):
        """
        Gráfico de barras comparando modelos
        """
        plt.figure(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(comparison_df)))
        
        plt.barh(comparison_df['Model'], comparison_df[metric], color=colors, 
                edgecolor='black', linewidth=1.5)
        plt.xlabel(metric, fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Anotar valores
        for i, (idx, row) in enumerate(comparison_df.iterrows()):
            plt.text(row[metric], i, f" {row[metric]:.4f}", 
                    va='center', fontsize=10, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def analyze_errors(self, y_true, y_pred, percentile=95):
        """
        Análisis detallado de errores
        """
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        
        print("\n" + "="*60)
        print("ANÁLISIS DE ERRORES")
        print("="*60)
        print(f"\nResidual Statistics:")
        print(f"  Mean:   {residuals.mean():.4f}")
        print(f"  Median: {np.median(residuals):.4f}")
        print(f"  Std:    {residuals.std():.4f}")
        print(f"\nAbsolute Error Statistics:")
        print(f"  Mean:   {abs_residuals.mean():.4f}")
        print(f"  Median: {np.median(abs_residuals):.4f}")
        print(f"  Max:    {abs_residuals.max():.4f}")
        print(f"  {percentile}th percentile: {np.percentile(abs_residuals, percentile):.4f}")
        
        # Identificar predicciones con mayor error
        error_threshold = np.percentile(abs_residuals, percentile)
        high_error_mask = abs_residuals > error_threshold
        
        print(f"\nHigh Error Predictions (>{percentile}th percentile):")
        print(f"  Count: {high_error_mask.sum()}")
        print(f"  Percentage: {100 * high_error_mask.sum() / len(y_true):.2f}%")
        
        return {
            'residuals': residuals,
            'abs_residuals': abs_residuals,
            'high_error_mask': high_error_mask
        }
    
    def save_results_to_csv(self, filepath='model_results.csv'):
        """Guarda resultados en CSV"""
        if not self.results:
            print("⚠️  No hay resultados para guardar")
            return
        
        df = pd.DataFrame(self.results).T
        df.to_csv(filepath, index=False)
        print(f"✅ Results saved to {filepath}")


def create_kaggle_submission(test_ids, predictions, filename='submission.csv'):
    """
    Crea archivo de submission para Kaggle
    
    Args:
        test_ids: IDs del test set
        predictions: predicciones (ya en escala original, no log)
        filename: nombre del archivo
    """
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"\n✅ Submission file created: {filename}")
    print(f"   Shape: {submission.shape}")
    print(f"   Sample:\n{submission.head()}")
    
    return submission


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("-" * 40)
    print("Usage:")
    print("  evaluator = ModelEvaluator()")
    print("  evaluator.print_metrics(y_true, y_pred, 'XGBoost')")
    print("  evaluator.plot_predictions_vs_actual(y_true, y_pred)")