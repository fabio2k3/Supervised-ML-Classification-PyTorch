"""
Módulo de preprocesamiento y feature engineering para House Prices dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class HousePricePreprocessor:
    """
    Preprocesador completo para el dataset de House Prices
    Incluye: limpieza, imputación, feature engineering, encoding y escalado
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_log_transformed = False
        
    def load_data(self, train_path, test_path=None):
        """Carga los datasets de entrenamiento y prueba"""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path else None
        
        print(f"✅ Train shape: {train_df.shape}")
        if test_df is not None:
            print(f"✅ Test shape: {test_df.shape}")
        
        return train_df, test_df
    
    def handle_missing_values(self, df):
        """Manejo robusto de valores faltantes"""
        df = df.copy()
        
        # Variables donde NA significa "None" o "No tiene"
        none_vars = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                     'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                     'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                     'BsmtFinType2', 'MasVnrType']
        
        for var in none_vars:
            if var in df.columns:
                df[var].fillna('None', inplace=True)
        
        # Variables numéricas relacionadas con garage/basement
        zero_vars = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                     'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                     'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
        
        for var in zero_vars:
            if var in df.columns:
                df[var].fillna(0, inplace=True)
        
        # LotFrontage: imputar con mediana por vecindario
        if 'LotFrontage' in df.columns:
            df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Resto de variables categóricas: moda
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Resto de variables numéricas: mediana
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        print(f"✅ Missing values handled. Remaining: {df.isnull().sum().sum()}")
        return df
    
    def remove_outliers(self, df, target_col='SalePrice'):
        """Elimina outliers extremos basados en análisis del dataset"""
        if target_col not in df.columns:
            return df
        
        df = df.copy()
        initial_shape = df.shape[0]
        
        # Outliers conocidos del dataset de House Prices
        df = df.drop(df[(df['GrLivArea'] > 4000) & (df[target_col] < 300000)].index)
        df = df.drop(df[(df['TotalBsmtSF'] > 6000)].index)
        
        print(f"✅ Outliers removed: {initial_shape - df.shape[0]} rows")
        return df
    
    def create_features(self, df):
        """Feature Engineering: creación de nuevas características"""
        df = df.copy()
        
        # 1. TotalSF: Total de pies cuadrados
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        
        # 2. Total de baños
        df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                           df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
        
        # 3. Edad de la casa
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        
        # 4. Indicador de remodelación
        df['WasRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
        
        # 5. Calidad total (combinación de métricas de calidad)
        df['TotalQual'] = df['OverallQual'] + df['OverallCond']
        
        # 6. Área total de porches
        df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                              df['3SsnPorch'] + df['ScreenPorch'])
        
        # 7. Indicador de piscina
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        
        # 8. Indicador de garage
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        
        # 9. Indicador de basement
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
        
        # 10. Indicador de segunda planta
        df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
        
        print(f"✅ Feature engineering completed. New shape: {df.shape}")
        return df
    
    def encode_categorical(self, df, target_col='SalePrice'):
        """Encoding de variables categóricas y ordinales"""
        df = df.copy()
        
        # Variables ordinales con orden específico
        ordinal_mappings = {
            'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'PoolQC': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
            'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
            'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
        }
        
        # Aplicar mapeos ordinales
        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # One-hot encoding para variables nominales
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != target_col]
        
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        print(f"✅ Encoding completed. Shape: {df.shape}")
        return df
    
    def transform_target(self, y):
        """Transformación logarítmica del target"""
        self.target_log_transformed = True
        return np.log1p(y)
    
    def inverse_transform_target(self, y_log):
        """Inversión de la transformación logarítmica"""
        return np.expm1(y_log)
    
    def scale_features(self, X_train, X_test=None):
        """Escalado de features numéricas"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def preprocess_pipeline(self, train_df, test_df=None, target_col='SalePrice'):
        """
        Pipeline completo de preprocesamiento
        
        Returns:
            X_train, y_train, X_test (si existe), test_ids
        """
        print("\n" + "="*60)
        print("INICIANDO PIPELINE DE PREPROCESAMIENTO")
        print("="*60 + "\n")
        
        # Guardar IDs
        train_ids = train_df['Id'] if 'Id' in train_df.columns else None
        test_ids = test_df['Id'] if test_df is not None and 'Id' in test_df.columns else None
        
        # Separar target
        y_train = None
        if target_col in train_df.columns:
            y_train = train_df[target_col].copy()
            train_df = train_df.drop(columns=[target_col])
        
        # Eliminar columna Id
        if 'Id' in train_df.columns:
            train_df = train_df.drop(columns=['Id'])
        if test_df is not None and 'Id' in test_df.columns:
            test_df = test_df.drop(columns=['Id'])
        
        # 1. Remover outliers (solo train)
        if y_train is not None:
            combined = pd.concat([train_df, pd.DataFrame({target_col: y_train})], axis=1)
            combined = self.remove_outliers(combined, target_col)
            train_df = combined.drop(columns=[target_col])
            y_train = combined[target_col]
        
        # 2. Concatenar train y test para procesamiento consistente
        if test_df is not None:
            n_train = len(train_df)
            combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        else:
            combined_df = train_df.copy()
        
        # 3. Missing values
        combined_df = self.handle_missing_values(combined_df)
        
        # 4. Feature engineering
        combined_df = self.create_features(combined_df)
        
        # 5. Encoding
        combined_df = self.encode_categorical(combined_df)
        
        # 6. Separar train y test
        if test_df is not None:
            train_processed = combined_df.iloc[:n_train, :]
            test_processed = combined_df.iloc[n_train:, :]
        else:
            train_processed = combined_df
            test_processed = None
        
        # 7. Transformar target (log)
        if y_train is not None:
            y_train_log = self.transform_target(y_train)
            print(f"✅ Target transformed: log(SalePrice)")
        else:
            y_train_log = None
        
        # 8. Guardar nombres de features
        self.feature_names = train_processed.columns.tolist()
        
        print("\n" + "="*60)
        print("PREPROCESAMIENTO COMPLETADO")
        print("="*60)
        print(f"Train shape: {train_processed.shape}")
        if test_processed is not None:
            print(f"Test shape: {test_processed.shape}")
        print(f"Features: {len(self.feature_names)}")
        
        return train_processed, y_train_log, test_processed, test_ids