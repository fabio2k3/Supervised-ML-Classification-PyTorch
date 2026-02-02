"""
Modelos de Ensemble: Bagging y Boosting
Stage 3: Implementing Advanced Ensemble Methods
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost no disponible. Instalar con: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM no disponible. Instalar con: pip install lightgbm")


class EnsembleModels:
    """
    Clase para modelos de ensemble: Bagging y Boosting
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """Inicializa los modelos de ensemble"""
        
        # BAGGING: Random Forest
        self.models['RandomForest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        # BOOSTING: XGBoost
        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
        
        # BOOSTING: LightGBM
        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        print(f"✅ {len(self.models)} ensemble models initialized")
        return self.models
        
    def evaluate_model(self, model, X, y, cv=5):
        """
        Evalúa un modelo usando cross-validation
        
        Returns:
            dict: métricas RMSE y MAE
        """
        # RMSE
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
        Entrena y evalúa todos los modelos de ensemble
        
        Returns:
            dict: resultados de todos los modelos
        """
        print("\n" + "="*60)
        print("EVALUACIÓN DE MODELOS ENSEMBLE")
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
                'Type': 'Bagging' if 'Forest' in model_name else 'Boosting',
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
    
    def get_feature_importance(self, model_name, feature_names=None):
        """
        Obtiene la importancia de features del modelo
        
        Returns:
            DataFrame con features ordenadas por importancia
        """
        import pandas as pd
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        else:
            print(f"⚠️  {model_name} no tiene atributo feature_importances_")
            return None
    
    def predict(self, model_name, X):
        """Genera predicciones con un modelo específico"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)


if __name__ == "__main__":
    # Verificar disponibilidad de librerías
    print("Ensemble Models Module - Status Check")
    print("-" * 40)
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
    print("-" * 40)
    print("\nUsage:")
    print("  ensemble = EnsembleModels()")
    print("  ensemble.initialize_models()")
    print("  results = ensemble.train_and_evaluate_all(X_train, y_train)")