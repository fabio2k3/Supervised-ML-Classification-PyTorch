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
    Stacking Ensemble implementation for regression

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
        """Initialize base estimators and meta-learner"""

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

        # Add XGBoost if available
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

        # Add LightGBM if available
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
        Train the stacking model

        Process:
        1. Generate out-of-fold predictions for each base model
        2. Use these predictions as features for the meta-model
        3. Train base models on the full training set
        4. Train meta-model using out-of-fold predictions
        """
        if len(self.base_models) == 0:
            self.initialize_models()

        print("\n" + "=" * 60)
        print("TRAINING STACKING ENSEMBLE")
        print("=" * 60 + "\n")

        n_models = len(self.base_models)
        n_samples = X_train.shape[0]

        # Matrix to store out-of-fold predictions
        self.base_predictions_train = np.zeros((n_samples, n_models))

        # Step 1: Generate out-of-fold predictions for each base model
        print("Generating out-of-fold predictions from base models...")
        for i, (name, model) in enumerate(self.base_models):
            print(f"  {i + 1}/{n_models}: {name}...")

            # Cross-validation predictions (out-of-fold)
            kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            oof_predictions = cross_val_predict(
                model, X_train, y_train,
                cv=kfold,
                n_jobs=-1
            )

            self.base_predictions_train[:, i] = oof_predictions

            # Train model on full dataset
            model.fit(X_train, y_train)

        print("\n✅ Base models trained")

        # Step 2: Train meta-model using out-of-fold predictions
        print("\nTraining meta-learner...")
        self.meta_model.fit(self.base_predictions_train, y_train)
        print("✅ Meta-learner trained")

        # Training metrics (using out-of-fold predictions)
        final_predictions = self.meta_model.predict(self.base_predictions_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, final_predictions))
        train_mae = mean_absolute_error(y_train, final_predictions)

        print(f"\nTraining metrics (out-of-fold):")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  MAE:  {train_mae:.4f}")

        return self

    def predict(self, X_test):
        """
        Generate predictions using stacking

        Process:
        1. Each base model generates predictions on X_test
        2. Meta-model combines these predictions to produce final output
        """
        if self.meta_model is None:
            raise ValueError("Model not trained. Call fit() first.")

        n_models = len(self.base_models)
        n_samples = X_test.shape[0]

        # Matrix for base model predictions
        base_predictions_test = np.zeros((n_samples, n_models))

        # Generate predictions from each base model
        for i, (name, model) in enumerate(self.base_models):
            base_predictions_test[:, i] = model.predict(X_test)

        # Final prediction from meta-model
        final_predictions = self.meta_model.predict(base_predictions_test)

        return final_predictions

    def evaluate(self, X_val, y_val):
        """Evaluate model on a validation set"""
        predictions = self.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)

        return {
            'RMSE': rmse,
            'MAE': mae
        }

    def get_base_model_predictions(self, X):
        """
        Return individual predictions from each base model
        Useful for analysis
        """
        predictions = {}
        for name, model in self.base_models:
            predictions[name] = model.predict(X)

        return predictions

    def get_meta_weights(self):
        """
        Return meta-learner coefficients
        Indicates the weight assigned to each base model
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
    Alternative implementation: Blending
    Similar to stacking but uses a validation set instead of CV
    """

    def __init__(self, random_state=42, val_size=0.2):
        self.random_state = random_state
        self.val_size = val_size
        self.base_models = []
        self.meta_model = None

    def initialize_models(self):
        """Initialize base estimators and meta-learner"""
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
        """Train using blending (train/validation split)"""
        from sklearn.model_selection import train_test_split

        if len(self.base_models) == 0:
            self.initialize_models()

        # Split into training and validation sets
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=self.val_size,
            random_state=self.random_state
        )

        n_models = len(self.base_models)
        val_predictions = np.zeros((X_val.shape[0], n_models))

        # Train base models on training set
        for i, (name, model) in enumerate(self.base_models):
            model.fit(X_tr, y_tr)
            val_predictions[:, i] = model.predict(X_val)

        # Train meta-model on validation set
        self.meta_model.fit(val_predictions, y_val)

        # Retrain base models on full dataset
        for name, model in self.base_models:
            model.fit(X_train, y_train)

        return self

    def predict(self, X_test):
        """Generate predictions using blending"""
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
