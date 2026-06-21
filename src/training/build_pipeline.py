from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def build_pipeline(n_estimators: int, max_depth: int, random_state: int) -> Pipeline:
    """Builds a complete scikit-learn training pipeline including preprocessing and classifier.

    Args:
        n_estimators (int): Number of trees in the RandomForest.
        max_depth (int): Max depth of the trees.
        random_state (int): Random seed for reproducibility.

    Returns:
        Pipeline: Complete scikit-learn Pipeline instance.

    Dependencies:
        - sklearn.compose.ColumnTransformer
        - sklearn.preprocessing.StandardScaler, OneHotEncoder
        - sklearn.pipeline.Pipeline
        - sklearn.ensemble.RandomForestClassifier
    """
    numerical_cols = ['MONTH', 'DAY_OF_WEEK', 'DISTANCE', 'DEPARTURE_DELAY']
    categorical_cols = ['AIRLINE']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    return pipeline
