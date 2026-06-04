import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None

    def build_pipeline(self):
        # REMOVED 'delay_minutes' to prevent fatal data leakage
        numeric_features = ['distance_km', 'parcel_weight_kg', 'parcel_value_sgd', 
                            'num_stops_on_route', 'driver_experience_months']
        
        categorical_features = ['branch', 'parcel_category', 'delivery_priority', 
                                'vehicle_type', 'payment_method']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        return self.preprocessor

    def fit_transform(self, X: pd.DataFrame):
        if self.preprocessor is None:
            self.build_pipeline()
        X_processed = self.preprocessor.fit_transform(X)
        
        # Robust feature name extraction (sklearn >= 1.0)
        raw_names = list(self.preprocessor.get_feature_names_out())
        # Clean up sklearn prefixes (e.g., 'num__distance_km' -> 'distance_km')
        self.feature_names = [name.replace("num__", "").replace("cat__", "") for name in raw_names]
        return X_processed

    def transform(self, X: pd.DataFrame):
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted first using fit_transform().")
        return self.preprocessor.transform(X)