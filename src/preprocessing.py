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
        numeric_features = ['distance_km', 'parcel_weight_kg', 'parcel_value_sgd', 
                            'num_stops_on_route', 'driver_experience_months']
        
        # ✅ Fixed typos here
        categorical_features = ['branch', 'parcel_category', 'delivery_priority', 
                                'vehicle_type', 'payment_method']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # ✅ Fixed typo & modern sparse_output flag
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        return self.preprocessor

    def fit_transform(self, X: pd.DataFrame):
        if self.preprocessor is None:
            self.build_pipeline()
        X_processed = self.preprocessor.fit_transform(X)
        
        # Extract & clean feature names for SHAP compatibility
        raw_names = list(self.preprocessor.get_feature_names_out())
        self.feature_names = [name.replace("num__", "").replace("cat__", "") for name in raw_names]
        return X_processed

    def transform(self, X: pd.DataFrame):
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted first using fit_transform().")
        return self.preprocessor.transform(X)