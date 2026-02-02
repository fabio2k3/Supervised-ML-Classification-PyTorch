"""
Stacking Ensemble Model
Stage 3-4: Stacked Generalization with Meta-learner
"""

import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class StackingRegressor:
    """
    Implementación de Stacking Ensemble para regresión
    
    Base estimators: Lasso, Random Forest, XGBoost
    Meta-learner: Ridge Regression
    """
    
    def __init__(self, random_state=42, cv=5):
        self.random_state = random_state
        self.cv = cv
        self.base_models = []
        self.meta_model = None
        self.base_predictions_train = None
        
    def initialize_models(self):
        """Inicializa base estimators y meta-learner"""
        
        # BASE ESTIMATORS
        base_models = [
            ('Lasso', Lasso(alpha=0.001, random_state=self.random_state, max_iter=10000)),
            ('RandomForest', RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ]
        
        # Agregar XGBoost si está disponible
        if XGBOOST_AVAILABLE:
            base_models.append(
                ('XGBoost', xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0
                ))
            )
        
        # Agregar LightGBM si está disponible
        if LIGHTGBM_AVAILABLE:
            base_models.append(
                ('LightGBM', lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                ))
            )
        
        self.base_models = base_models
        
        # META-LEARNER: Ridge Regression
        self.meta_model = Ridge(alpha=10.0, random_state=self.random_state)
        
        print(f"✅ Stacking initialized with {len(self.base_models)} base models")
        print(f"   Base models: {[name for name, _ in self.base_models]}")
        print(f"   Meta-learner: Ridge Regression")
        
    def fit(self, X_train, y_train):
        """
        Entrena el modelo de stacking
        
        Proceso:
        1. Genera predicciones out-of-fold de cada base model
        2. Usa estas predicciones como features para el meta-model
        3. Entrena base models en todo el training set
        4. Entrena meta-model con las predicciones out-of-fold
        """
        if len(self.base_models) == 0:
            self.initialize_models()
        
        print("\n" + "="*60)
        print("ENTRENANDO STACKING ENSEMBLE")
        print("="*60 + "\n")
        
        n_models = len(self.base_models)
        n_samples = X_train.shape[0]
        
        # Matriz para almacenar predicciones out-of-fold
        self.base_predictions_train = np.zeros((n_samples, n_models))
        
        # Paso 1: Generar predicciones out-of-fold para cada base model
        print("Generando predicciones out-of-fold de base models...")
        for i, (name, model) in enumerate(self.base_models):
            print(f"  {i+1}/{n_models}: {name}...")
            
            # Cross-validation predictions (out-of-fold)
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            oof_predictions = cross_val_predict(
                model, X_train, y_train, 
                cv=kfold, 
                n_jobs=-1
            )
            
            self.base_predictions_train[:, i] = oof_predictions
            
            # Entrenar en todo el dataset
            model.fit(X_train, y_train)
        
        print("\n✅ Base models entrenados")
        
        # Paso 2: Entrenar meta-model con las predicciones out-of-fold
        print("\nEntrenando meta-learner...")
        self.meta_model.fit(self.base_predictions_train, y_train)
        print("✅ Meta-learner entrenado")
        
        # Calcular métricas en training (usando predicciones out-of-fold)
        final_predictions = self.meta_model.predict(self.base_predictions_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, final_predictions))
        train_mae = mean_absolute_error(y_train, final_predictions)
        
        print(f"\nMétricas de training (out-of-fold):")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE:  {train_mae:.4f}")
        
        return self
    
    def predict(self, X_test):
        """
        Genera predicciones usando stacking
        
        Proceso:
        1. Cada base model genera predicciones en X_test
        2. Meta-model combina estas predicciones para dar la predicción final
        """
        if self.meta_model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        n_models = len(self.base_models)
        n_samples = X_test.shape[0]
        
        # Matriz para predicciones de base models
        base_predictions_test = np.zeros((n_samples, n_models))
        
        # Generar predicciones con cada base model
        for i, (name, model) in enumerate(self.base_models):
            base_predictions_test[:, i] = model.predict(X_test)
        
        # Meta-model hace la predicción final
        final_predictions = self.meta_model.predict(base_predictions_test)
        
        return final_predictions
    
    def evaluate(self, X_val, y_val):
        """Evalúa el modelo en un conjunto de validación"""
        predictions = self.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)
        
        return {
            'RMSE': rmse,
            'MAE': mae
        }
    
    def get_base_model_predictions(self, X):
        """
        Retorna las predicciones individuales de cada base model
        Útil para análisis
        """
        predictions = {}
        for name, model in self.base_models:
            predictions[name] = model.predict(X)
        
        return predictions
    
    def get_meta_weights(self):
        """
        Retorna los coeficientes del meta-learner
        Indica qué peso le da a cada base model
        """
        if hasattr(self.meta_model, 'coef_'):
            weights = {}
            for i, (name, _) in enumerate(self.base_models):
                weights[name] = self.meta_model.coef_[i]
            return weights
        else:
            return None


class BlendingRegressor:
    """
    Implementación alternativa: Blending
    Similar a Stacking pero usa un validation set en vez de CV
    """
    
    def __init__(self, random_state=42, val_size=0.2):
        self.random_state = random_state
        self.val_size = val_size
        self.base_models = []
        self.meta_model = None
        
    def initialize_models(self):
        """Inicializa base estimators y meta-learner"""
        from sklearn.linear_model import Lasso
        from sklearn.ensemble import RandomForestRegressor
        
        self.base_models = [
            ('Lasso', Lasso(alpha=0.001, random_state=self.random_state, max_iter=10000)),
            ('RandomForest', RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ]
        
        if XGBOOST_AVAILABLE:
            self.base_models.append(
                ('XGBoost', xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0
                ))
            )
        
        self.meta_model = Ridge(alpha=10.0, random_state=self.random_state)
        
        print(f"✅ Blending initialized with {len(self.base_models)} base models")
    
    def fit(self, X_train, y_train):
        """Entrena usando blending (train/val split)"""
        from sklearn.model_selection import train_test_split
        
        if len(self.base_models) == 0:
            self.initialize_models()
        
        # Split en train y validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, 
            test_size=self.val_size, 
            random_state=self.random_state
        )
        
        n_models = len(self.base_models)
        val_predictions = np.zeros((X_val.shape[0], n_models))
        
        # Entrenar base models en training set
        for i, (name, model) in enumerate(self.base_models):
            model.fit(X_tr, y_tr)
            val_predictions[:, i] = model.predict(X_val)
        
        # Entrenar meta-model en validation set
        self.meta_model.fit(val_predictions, y_val)
        
        # Re-entrenar base models en todo el dataset
        for name, model in self.base_models:
            model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X_test):
        """Genera predicciones usando blending"""
        n_models = len(self.base_models)
        base_predictions = np.zeros((X_test.shape[0], n_models))
        
        for i, (name, model) in enumerate(self.base_models):
            base_predictions[:, i] = model.predict(X_test)
        
        return self.meta_model.predict(base_predictions)


if __name__ == "__main__":
    print("Stacking Module - Status Check")
    print("-" * 40)
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
    print("-" * 40)
    print("\nUsage:")
    print("  stacking = StackingRegressor()")
    print("  stacking.fit(X_train, y_train)")
    print("  predictions = stacking.predict(X_test)")