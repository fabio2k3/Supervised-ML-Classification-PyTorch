"""
Modelos base para benchmarking
Stage 2: Individual Base Model Benchmarking
"""

import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class BaseModels:
    """
    Clase para entrenar y evaluar modelos base
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """Inicializa los tres modelos base requeridos"""
        self.models = {
            'Lasso': Lasso(alpha=0.001, random_state=self.random_state, max_iter=10000),
            'Ridge': Ridge(alpha=10.0, random_state=self.random_state),
            'KNN': KNeighborsRegressor(n_neighbors=5, weights='distance'),
            'DecisionTree': DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state
            )
        }
        print(f"✅ {len(self.models)} base models initialized")
        
    def evaluate_model(self, model, X, y, cv=5):
        """
        Evalúa un modelo usando cross-validation
        
        Returns:
            dict: métricas RMSE y MAE
        """
        # RMSE (negativo porque sklearn usa negative MSE)
        mse_scores = -cross_val_score(
            model, X, y, 
            cv=cv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rmse_scores = np.sqrt(mse_scores)
        
        # MAE
        mae_scores = -cross_val_score(
            model, X, y,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        return {
            'RMSE_mean': rmse_scores.mean(),
            'RMSE_std': rmse_scores.std(),
            'MAE_mean': mae_scores.mean(),
            'MAE_std': mae_scores.std()
        }
    
    def train_and_evaluate_all(self, X_train, y_train, cv=5):
        """
        Entrena y evalúa todos los modelos base
        
        Returns:
            dict: resultados de todos los modelos
        """
        print("\n" + "="*60)
        print("EVALUACIÓN DE MODELOS BASE")
        print("="*60 + "\n")
        
        if len(self.models) == 0:
            self.initialize_models()
        
        for name, model in self.models.items():
            print(f"Evaluando {name}...")
            
            # Cross-validation
            metrics = self.evaluate_model(model, X_train, y_train, cv=cv)
            
            # Entrenar en todo el dataset
            model.fit(X_train, y_train)
            
            # Guardar resultados
            self.results[name] = metrics
            
            print(f"  RMSE: {metrics['RMSE_mean']:.4f} (+/- {metrics['RMSE_std']:.4f})")
            print(f"  MAE:  {metrics['MAE_mean']:.4f} (+/- {metrics['MAE_std']:.4f})\n")
        
        return self.results
    
    def get_results_table(self):
        """Retorna tabla de resultados formateada"""
        import pandas as pd
        
        results_data = []
        for model_name, metrics in self.results.items():
            results_data.append({
                'Model': model_name,
                'RMSE (mean)': f"{metrics['RMSE_mean']:.4f}",
                'RMSE (std)': f"{metrics['RMSE_std']:.4f}",
                'MAE (mean)': f"{metrics['MAE_mean']:.4f}",
                'MAE (std)': f"{metrics['MAE_std']:.4f}"
            })
        
        return pd.DataFrame(results_data)
    
    def get_best_model(self):
        """Retorna el mejor modelo según RMSE"""
        if not self.results:
            return None
        
        best_name = min(self.results.keys(), 
                       key=lambda x: self.results[x]['RMSE_mean'])
        return best_name, self.models[best_name]
    
    def predict(self, model_name, X):
        """Genera predicciones con un modelo específico"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)


if __name__ == "__main__":
    # Ejemplo de uso
    print("Base Models Module - Ready to use")
    print("Usage:")
    print("  base_models = BaseModels()")
    print("  base_models.initialize_models()")
    print("  results = base_models.train_and_evaluate_all(X_train, y_train)")