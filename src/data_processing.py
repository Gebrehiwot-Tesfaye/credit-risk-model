import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer


# 1. Load data

def load_data(filepath):
    """Load CSV data into a pandas DataFrame."""
    return pd.read_csv(filepath)


# 2. Data overview

def data_overview(df):
    print('Shape:', df.shape)
    print('\nData types:')
    print(df.dtypes)
    print('\nFirst 5 rows:')
    print(df.head())


# 3. Summary statistics

def summary_stats(df):
    print(df.describe(include='all').transpose())


# 4. Distribution of numerical features

def plot_numerical_distributions(df, num_cols, bins=30):
    df[num_cols].hist(bins=bins, figsize=(15, 8), layout=(-1, 3))
    plt.tight_layout()
    plt.show()


# 5. Distribution of categorical features

def plot_categorical_distributions(df, cat_cols, max_unique=40, top_n=20):
    """
    Plots bar charts for all categorical columns.
    For columns with >max_unique unique values, only the top_n most frequent are shown.
    """
    n = len(cat_cols)
    fig, axes = plt.subplots(
    (n + 2) // 3, 3,
    figsize=(15, 5 * ((n + 2) // 3))
)
    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        value_counts = df[col].value_counts()
        if len(value_counts) > max_unique:
            value_counts = value_counts[:top_n]
            axes[i].set_title(f"{col} (top {top_n})")
        else:
            axes[i].set_title(col)
        value_counts.plot(kind='bar', ax=axes[i])
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


# 6. Correlation analysis

def plot_correlation_matrix(df, num_cols):
    corr = df[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


# 7. Missing values

def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * mis_val / len(df)
    mis_table = pd.DataFrame({
        'Missing Values': mis_val,
        '% of Total Values': mis_val_percent
    })
    mis_table = mis_table[mis_table['Missing Values'] > 0]
    mis_table = mis_table.sort_values(
        '% of Total Values', ascending=False
    )
    print(mis_table)
    return mis_table


# 8. Outlier detection (boxplots)

def plot_boxplots(df, num_cols):
    n = len(num_cols)
    fig, axes = plt.subplots((n + 2) // 3, 3,
                             figsize=(15, 5 * ((n + 2) // 3)))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()


# --- Feature Engineering Classes ---

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Create aggregate features per customer (AccountId).
    """
    def __init__(self, group_col='AccountId'):
        self.group_col = group_col
        self.agg_df = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby(self.group_col)['Amount'].agg([
            ('total_transaction_amount', 'sum'),
            ('avg_transaction_amount', 'mean'),
            ('transaction_count', 'count'),
            ('std_transaction_amount', 'std')
        ]).reset_index()
        X = X.merge(agg, on=self.group_col, how='left')
        return X

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract datetime features from TransactionStartTime.
    """
    def __init__(self, col='TransactionStartTime'):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.col] = pd.to_datetime(X[self.col], errors='coerce')
        X['transaction_hour'] = X[self.col].dt.hour
        X['transaction_day'] = X[self.col].dt.day
        X['transaction_month'] = X[self.col].dt.month
        X['transaction_year'] = X[self.col].dt.year
        return X

# --- Feature Engineering Pipeline ---

def get_feature_engineering_pipeline(
    num_cols, cat_cols, group_col='AccountId',
    scaling='standard', encoding='onehot', impute_strategy='mean'):
    """
    Returns a sklearn Pipeline for feature engineering.
    """
    # Imputation
    num_imputer = SimpleImputer(strategy=impute_strategy)
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Scaling
    scaler = StandardScaler() if scaling == 'standard' else MinMaxScaler()

    # Encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) if encoding == 'onehot' else 'passthrough'

    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', num_imputer),
            ('scaler', scaler)
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', cat_imputer),
            ('encoder', encoder)
        ]), cat_cols)
    ])

    # Full pipeline
    pipeline = Pipeline([
        ('datetime_features', DateTimeFeatures()),
        ('aggregate_features', AggregateFeatures(group_col=group_col)),
        ('preprocessor', preprocessor)
    ])
    return pipeline


