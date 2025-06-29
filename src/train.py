import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import mlflow
import mlflow.sklearn
from data_processing import load_data, get_feature_engineering_pipeline, add_is_high_risk


def prepare_data():
    """Load and prepare data with feature engineering and target variable."""
    # Load data
    df = load_data('data/raw/data.csv')
    
    # Add proxy target variable
    df = add_is_high_risk(df)
    
    # Define features
    num_cols = ['Amount', 'Value']
    cat_cols = [
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
        'ProductCategory', 'ChannelId', 'PricingStrategy'
    ]
    
    # Get feature engineering pipeline
    pipeline = get_feature_engineering_pipeline(num_cols, cat_cols)
    
    # Prepare features and target
    X = df.drop(['is_high_risk'], axis=1, errors='ignore')
    y = df['is_high_risk']
    
    # Apply feature engineering
    X_transformed = pipeline.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, pipeline


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics."""
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


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression with hyperparameter tuning."""
    with mlflow.start_run(run_name="logistic_regression"):
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
        # Grid search
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test, "Logistic Regression")
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "logistic_regression")
        
        return grid_search.best_estimator_, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with hyperparameter tuning."""
    with mlflow.start_run(run_name="random_forest"):
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test, "Random Forest")
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")
        
        return grid_search.best_estimator_, metrics


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting with hyperparameter tuning."""
    with mlflow.start_run(run_name="gradient_boosting"):
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Grid search
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test, "Gradient Boosting")
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "gradient_boosting")
        
        return grid_search.best_estimator_, metrics


def register_best_model(best_model, best_metrics, model_name):
    """Register the best model in MLflow Model Registry."""
    with mlflow.start_run(run_name=f"best_model_{model_name}"):
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(best_model, "best_model")
        
        # Register model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
        mlflow.register_model(model_uri, f"credit_risk_{model_name}")


def main():
    """Main training function."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, pipeline = prepare_data()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Target distribution: {np.bincount(y_train)}")
    
    # Train models
    print("\nTraining Logistic Regression...")
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    print("\nTraining Random Forest...")
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    
    print("\nTraining Gradient Boosting...")
    gb_model, gb_metrics = train_gradient_boosting(X_train, y_train, X_test, y_test)
    
    # Find best model
    models = {
        'logistic_regression': (lr_model, lr_metrics),
        'random_forest': (rf_model, rf_metrics),
        'gradient_boosting': (gb_model, gb_metrics)
    }
    
    best_model_name = max(models.keys(), key=lambda k: models[k][1]['f1_score'])
    best_model, best_metrics = models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best F1 Score: {best_metrics['f1_score']:.4f}")
    
    # Register best model
    register_best_model(best_model, best_metrics, best_model_name)
    
    print("\nTraining completed! Check MLflow UI for detailed results.")


if __name__ == "__main__":
    main()
