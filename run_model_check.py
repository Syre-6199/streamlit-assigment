import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

CSV = 'Airbnb_site_hotel new.csv'

def load():
    df = pd.read_csv(CSV)
    cols_to_drop = ['id', 'host_id', 'host_name', 'listingh number', 'listing number', 'listing_number']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    numeric_columns = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'total reviewers number', 'host total listings count']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['price'])
    df = df[df['price'] > 0]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def baseline_and_improved(df, features):
    X = df[features].copy()
    y = df['price'].copy()
    # remove outliers using IQR on y
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (y >= lower) & (y <= upper) & (y > 0)
    X = X[mask]
    y = y[mask]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline linear
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)
    r2_lin = r2_score(y_test, y_pred_lin)

    # Improved pipeline (log target)
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    pipeline = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(alpha=1.0))
    pipeline.fit(X_train, y_train_log)
    y_pred_log = pipeline.predict(X_test)
    y_pred_imp = np.expm1(y_pred_log)
    r2_imp = r2_score(y_test, y_pred_imp)

    print(f"Baseline Linear R2: {r2_lin:.4f}")
    print(f"Improved Pipeline R2: {r2_imp:.4f}")


if __name__ == '__main__':
    try:
        df = load()
    except Exception as e:
        print('Error loading CSV:', e)
        raise
    # determine available features
    feature_columns = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'total reviewers number', 'host total listings count']
    features = [c for c in feature_columns if c in df.columns]
    if not features:
        print('No numeric features found. Available columns:', df.columns.tolist())
    else:
        print('Using features:', features)
        baseline_and_improved(df, features)
