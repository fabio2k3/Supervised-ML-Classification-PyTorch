"""
Script principal de entrenamiento para House Prices Competition
Ejecuta el pipeline completo: preprocessing, modelos base, ensemble y stacking

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

# Agregar src al path
sys.path.append('src')

from preprocessing import HousePricePreprocessor
from base_models import BaseModels
from ensemble_models import EnsembleModels
from stacking import StackingRegressor
from evaluation import ModelEvaluator, create_kaggle_submission


def setup_directories():
    """Crea directorios necesarios para outputs"""
    directories = ['outputs', 'outputs/models', 'outputs/plots', 'outputs/submissions']
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Directorios configurados")


def load_and_preprocess_data(train_path='data/train.csv', test_path='data/test.csv'):
    """
    Carga y preprocesa los datos
    
    Returns:
        X_train, y_train, X_test, test_ids, preprocessor
    """
    print("\n" + "="*70)
    print("PASO 1: CARGA Y PREPROCESAMIENTO DE DATOS")
    print("="*70)
    
    preprocessor = HousePricePreprocessor()
    
    # Cargar datos
    train_df, test_df = preprocessor.load_data(train_path, test_path)
    
    # Aplicar pipeline completo
    X_train, y_train, X_test, test_ids = preprocessor.preprocess_pipeline(
        train_df, test_df, target_col='SalePrice'
    )
    
    # Convertir a numpy arrays si son DataFrames
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        X_train = X_train.values
    else:
        feature_names = preprocessor.feature_names
    
    if X_test is not None and isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    
    print(f"\n✅ Preprocesamiento completado")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    if X_test is not None:
        print(f"   X_test shape: {X_test.shape}")
    print(f"   Features: {len(feature_names)}")
    
    return X_train, y_train, X_test, test_ids, preprocessor, feature_names


def train_base_models(X_train, y_train, cv=5):
    """
    Entrena y evalúa modelos base
    
    Returns:
        BaseModels object, results dict
    """
    print("\n" + "="*70)
    print("PASO 2: ENTRENAMIENTO DE MODELOS BASE")
    print("="*70)
    
    base_models = BaseModels(random_state=42)
    base_models.initialize_models()
    
    # Entrenar y evaluar con cross-validation
    results = base_models.train_and_evaluate_all(X_train, y_train, cv=cv)
    
    # Mostrar tabla de resultados
    results_table = base_models.get_results_table()
    print("\n📊 RESULTADOS MODELOS BASE:")
    print(results_table.to_string(index=False))
    
    # Guardar resultados
    results_table.to_csv('outputs/base_models_results.csv', index=False)
    print("\n✅ Resultados guardados en outputs/base_models_results.csv")
    
    # Mejor modelo
    best_name, best_model = base_models.get_best_model()
    print(f"\n🏆 Mejor modelo base: {best_name}")
    print(f"   RMSE: {results[best_name]['RMSE_mean']:.4f}")
    
    return base_models, results


def train_ensemble_models(X_train, y_train, cv=5):
    """
    Entrena y evalúa modelos de ensemble
    
    Returns:
        EnsembleModels object, results dict
    """
    print("\n" + "="*70)
    print("PASO 3: ENTRENAMIENTO DE MODELOS ENSEMBLE")
    print("="*70)
    
    ensemble_models = EnsembleModels(random_state=42)
    ensemble_models.initialize_models()
    
    # Entrenar y evaluar
    results = ensemble_models.train_and_evaluate_all(X_train, y_train, cv=cv)
    
    # Mostrar tabla de resultados
    results_table = ensemble_models.get_results_table()
    print("\n📊 RESULTADOS MODELOS ENSEMBLE:")
    print(results_table.to_string(index=False))
    
    # Guardar resultados
    results_table.to_csv('outputs/ensemble_models_results.csv', index=False)
    print("\n✅ Resultados guardados en outputs/ensemble_models_results.csv")
    
    # Mejor modelo
    best_name, best_model = ensemble_models.get_best_model()
    print(f"\n🏆 Mejor modelo ensemble: {best_name}")
    print(f"   RMSE: {results[best_name]['RMSE_mean']:.4f}")
    
    return ensemble_models, results


def train_stacking_model(X_train, y_train, cv=5):
    """
    Entrena modelo de stacking
    
    Returns:
        StackingRegressor object
    """
    print("\n" + "="*70)
    print("PASO 4: ENTRENAMIENTO DE STACKING ENSEMBLE")
    print("="*70)
    
    stacking = StackingRegressor(random_state=42, cv=cv)
    stacking.initialize_models()
    stacking.fit(X_train, y_train)
    
    # Mostrar pesos del meta-learner
    weights = stacking.get_meta_weights()
    if weights:
        print("\n📊 PESOS DEL META-LEARNER (Ridge):")
        for model_name, weight in weights.items():
            print(f"   {model_name}: {weight:.4f}")
    
    return stacking


def compare_all_models(X_train, y_train, base_models, ensemble_models, stacking, cv=5):
    """
    Compara todos los modelos entrenados
    """
    print("\n" + "="*70)
    print("PASO 5: COMPARACIÓN FINAL DE TODOS LOS MODELOS")
    print("="*70)
    
    from sklearn.model_selection import cross_val_score
    
    all_results = []
    
    # Modelos base
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
    
    # Modelos ensemble
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
    
    # Stacking (usar predicciones out-of-fold ya calculadas)
    stacking_pred = stacking.meta_model.predict(stacking.base_predictions_train)
    stacking_rmse = np.sqrt(mean_squared_error(y_train, stacking_pred))
    
    all_results.append({
        'Model': 'Stacking',
        'Type': 'Meta-Ensemble',
        'RMSE_mean': stacking_rmse,
        'RMSE_std': 0.0  # No tenemos std para stacking directo
    })
    
    # Crear DataFrame y ordenar por RMSE
    comparison_df = pd.DataFrame(all_results)
    comparison_df = comparison_df.sort_values('RMSE_mean')
    
    print("\n📊 COMPARACIÓN FINAL (ordenado por RMSE):")
    print(comparison_df.to_string(index=False))
    
    # Guardar comparación
    comparison_df.to_csv('outputs/final_comparison.csv', index=False)
    print("\n✅ Comparación guardada en outputs/final_comparison.csv")
    
    # Mejor modelo overall
    best_model = comparison_df.iloc[0]
    print(f"\n🏆 MEJOR MODELO OVERALL: {best_model['Model']}")
    print(f"   Tipo: {best_model['Type']}")
    print(f"   RMSE: {best_model['RMSE_mean']:.4f} (+/- {best_model['RMSE_std']:.4f})")
    
    return comparison_df


def generate_predictions(X_test, test_ids, preprocessor, stacking, output_name='stacking'):
    """
    Genera predicciones para Kaggle submission
    """
    print("\n" + "="*70)
    print("PASO 6: GENERACIÓN DE PREDICCIONES PARA KAGGLE")
    print("="*70)
    
    if X_test is None:
        print("⚠️  No hay datos de test disponibles")
        return None
    
    # Predicciones en escala log
    predictions_log = stacking.predict(X_test)
    
    # Invertir transformación logarítmica
    predictions = preprocessor.inverse_transform_target(predictions_log)
    
    # Crear archivo de submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'outputs/submissions/{output_name}_{timestamp}.csv'
    
    submission = create_kaggle_submission(test_ids, predictions, filename)
    
    print(f"\n✅ Submission creado: {filename}")
    print(f"   Estadísticas de predicciones:")
    print(f"   - Min:    ${predictions.min():,.2f}")
    print(f"   - Max:    ${predictions.max():,.2f}")
    print(f"   - Mean:   ${predictions.mean():,.2f}")
    print(f"   - Median: ${np.median(predictions):,.2f}")
    
    return submission


def create_visualizations(X_train, y_train, base_models, ensemble_models, 
                         stacking, feature_names, preprocessor):
    """
    Genera visualizaciones de análisis
    """
    print("\n" + "="*70)
    print("PASO 7: GENERACIÓN DE VISUALIZACIONES")
    print("="*70)
    
    evaluator = ModelEvaluator()
    
    # Predicciones de stacking para visualización
    from sklearn.model_selection import cross_val_predict, KFold
    
    # Usar predicciones out-of-fold del stacking
    stacking_pred = stacking.meta_model.predict(stacking.base_predictions_train)
    
    # Invertir log transform para visualización
    y_train_original = preprocessor.inverse_transform_target(y_train)
    stacking_pred_original = preprocessor.inverse_transform_target(stacking_pred)
    
    print("\n📊 Generando gráficos...")
    
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
    
    # 3. Feature importance del mejor modelo de ensemble
    best_ensemble_name, best_ensemble_model = ensemble_models.get_best_model()
    if hasattr(best_ensemble_model, 'feature_importances_'):
        evaluator.plot_feature_importance(
            best_ensemble_model,
            feature_names,
            top_n=20,
            model_name=best_ensemble_name,
            save_path=f'outputs/plots/{best_ensemble_name.lower()}_feature_importance.png'
        )
    
    print("\n✅ Visualizaciones guardadas en outputs/plots/")


def save_training_summary(comparison_df, preprocessor):
    """
    Guarda resumen del entrenamiento
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = f"""
{'='*70}
RESUMEN DE ENTRENAMIENTO - HOUSE PRICES COMPETITION
{'='*70}

Fecha: {timestamp}

PREPROCESAMIENTO:
- Features originales: 79
- Features finales: {len(preprocessor.feature_names)}
- Target transformation: log(SalePrice)
- Missing values: Imputados
- Outliers: Removidos

MODELOS ENTRENADOS:
{comparison_df.to_string(index=False)}

MEJOR MODELO: {comparison_df.iloc[0]['Model']}
- RMSE: {comparison_df.iloc[0]['RMSE_mean']:.4f}
- Tipo: {comparison_df.iloc[0]['Type']}

ARCHIVOS GENERADOS:
- outputs/base_models_results.csv
- outputs/ensemble_models_results.csv
- outputs/final_comparison.csv
- outputs/plots/*.png
- outputs/submissions/*.csv

{'='*70}
    """
    
    with open('outputs/training_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\n✅ Resumen guardado en outputs/training_summary.txt")
    print(summary)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Train House Prices models')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['all', 'base_models', 'ensemble', 'stacking'],
                       help='Etapa de entrenamiento a ejecutar')
    parser.add_argument('--train', type=str, default='data/train.csv',
                       help='Path al archivo train.csv')
    parser.add_argument('--test', type=str, default='data/test.csv',
                       help='Path al archivo test.csv')
    parser.add_argument('--cv', type=int, default=5,
                       help='Número de folds para cross-validation')
    parser.add_argument('--no-viz', action='store_true',
                       help='No generar visualizaciones')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HOUSE PRICES - ADVANCED REGRESSION TECHNIQUES")
    print("Pipeline de Entrenamiento Completo")
    print("="*70)
    
    # Setup
    setup_directories()
    
    # Cargar y preprocesar
    X_train, y_train, X_test, test_ids, preprocessor, feature_names = load_and_preprocess_data(
        args.train, args.test
    )
    
    # Variables para almacenar modelos
    base_models_obj = None
    ensemble_models_obj = None
    stacking_obj = None
    
    # Ejecutar etapas según argumento
    if args.stage in ['all', 'base_models']:
        base_models_obj, base_results = train_base_models(X_train, y_train, args.cv)
    
    if args.stage in ['all', 'ensemble']:
        ensemble_models_obj, ensemble_results = train_ensemble_models(X_train, y_train, args.cv)
    
    if args.stage in ['all', 'stacking']:
        stacking_obj = train_stacking_model(X_train, y_train, args.cv)
    
    # Comparación final (solo si se ejecutaron todos)
    if args.stage == 'all':
        comparison_df = compare_all_models(
            X_train, y_train, 
            base_models_obj, 
            ensemble_models_obj, 
            stacking_obj, 
            args.cv
        )
        
        # Generar predicciones para Kaggle
        if X_test is not None:
            generate_predictions(X_test, test_ids, preprocessor, stacking_obj)
        
        # Visualizaciones
        if not args.no_viz:
            create_visualizations(
                X_train, y_train,
                base_models_obj,
                ensemble_models_obj,
                stacking_obj,
                feature_names,
                preprocessor
            )
        
        # Guardar resumen
        save_training_summary(comparison_df, preprocessor)
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print("\nPróximos pasos:")
    print("1. Revisar outputs/training_summary.txt")
    print("2. Analizar gráficos en outputs/plots/")
    print("3. Subir submission a Kaggle desde outputs/submissions/")
    print("\n")


if __name__ == "__main__":
    # Importar sklearn.metrics para evitar error
    from sklearn.metrics import mean_squared_error
    
    main()