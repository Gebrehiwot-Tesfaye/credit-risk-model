import pandas as pd
import numpy as np
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import mlflow
import mlflow.sklearn
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from data_processing import load_data, add_is_high_risk

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def prepare_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'raw', 'data.csv')
    print(f"Looking for data at: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    df = load_data(data_path)
    df = df.sample(frac=0.1, random_state=42)
    print(f"Data loaded successfully. Shape: {df.shape}")
    df = add_is_high_risk(df)
    print(f"Target variable added. Shape: {df.shape}")
    # Drop ID columns from features
    id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    feature_cols = [
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory',
        'ChannelId', 'Amount', 'Value', 'TransactionStartTime', 'PricingStrategy'
    ]
    X = df[feature_cols]
    y = df['is_high_risk']
    return X, y

def get_pipeline(classifier):
    cat_cols = ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'TransactionStartTime']
    num_cols = ['CountryCode', 'Amount', 'Value', 'PricingStrategy']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    return pipeline

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    return metrics

def train_with_gridsearch(X_train, y_train, X_test, y_test, classifier, param_grid, model_name):
    pipeline = get_pipeline(classifier)
    grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='f1', n_jobs=1)
    grid_search.fit(X_train, y_train)
    mlflow.log_params(grid_search.best_params_)
    metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test, model_name)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(grid_search.best_estimator_, model_name)
    return grid_search.best_estimator_, metrics

def register_best_model(best_model, best_metrics, model_name):
    with mlflow.start_run(run_name=f"best_model_{model_name}"):
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(best_model, "best_model")
        model_uri = f"runs:/" + mlflow.active_run().info.run_id + "/best_model"
        mlflow.register_model(model_uri, f"credit_risk_{model_name}")
        pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'fitted_pipeline.joblib')
        joblib.dump(best_model, pipeline_path)
        print(f"Fitted pipeline saved to {pipeline_path}")

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    print("Loading and preparing data...")
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Target distribution: {np.bincount(y_train)}")
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    print("="*50)
    lr_param_grid = {
        'classifier__C': [1],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs']
    }
    lr_model, lr_metrics = train_with_gridsearch(
        X_train, y_train, X_test, y_test,
        LogisticRegression(random_state=42, max_iter=200),
        lr_param_grid, "logistic_regression"
    )
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    rf_param_grid = {
        'classifier__n_estimators': [50],
        'classifier__max_depth': [10],
        'classifier__min_samples_split': [5]
    }
    rf_model, rf_metrics = train_with_gridsearch(
        X_train, y_train, X_test, y_test,
        RandomForestClassifier(random_state=42, n_jobs=1),
        rf_param_grid, "random_forest"
    )
    print("\n" + "="*50)
    print("Training Gradient Boosting...")
    print("="*50)
    gb_param_grid = {
        'classifier__n_estimators': [50],
        'classifier__learning_rate': [0.1],
        'classifier__max_depth': [3]
    }
    gb_model, gb_metrics = train_with_gridsearch(
        X_train, y_train, X_test, y_test,
        GradientBoostingClassifier(random_state=42),
        gb_param_grid, "gradient_boosting"
    )
    models = {
        'logistic_regression': (lr_model, lr_metrics),
        'random_forest': (rf_model, rf_metrics),
        'gradient_boosting': (gb_model, gb_metrics)
    }
    best_model_name = max(models.keys(), key=lambda k: models[k][1]['f1_score'])
    best_model, best_metrics = models[best_model_name]
    print("\n" + "="*50)
    print(f"Best model: {best_model_name}")
    print(f"Best F1 Score: {best_metrics['f1_score']:.4f}")
    print("="*50)
    print("Registering best model in MLflow...")
    register_best_model(best_model, best_metrics, best_model_name)
    print("\n🎉 Training completed successfully!")
    print("You can now start the API with: uvicorn src.api.main:app --reload")

if __name__ == "__main__":
    main()
