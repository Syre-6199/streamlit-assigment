import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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


def eval_models(df, features):
    X = df[features].copy()
    y = df['price'].copy()
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (y >= lower) & (y <= upper) & (y > 0)
    X = X[mask]
    y = y[mask]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # Linear
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    results['Linear'] = r2_score(y_test, lin.predict(X_test))

    # Improved pipeline (log target)
    y_train_log = np.log1p(y_train)
    pipeline = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(alpha=1.0))
    pipeline.fit(X_train, y_train_log)
    y_pred_imp = np.expm1(pipeline.predict(X_test))
    results['Polynomial+Ridge (log-target)'] = r2_score(y_test, y_pred_imp)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results['RandomForest'] = r2_score(y_test, rf.predict(X_test))

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    results['GradientBoosting'] = r2_score(y_test, gb.predict(X_test))

    return results

if __name__ == '__main__':
    df = load()
    feature_columns = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'total reviewers number', 'host total listings count']
    features = [c for c in feature_columns if c in df.columns]
    if not features:
        print('No numeric features found. Available columns:', df.columns.tolist())
    else:
        print('Using features:', features)
        res = eval_models(df, features)
        for k, v in res.items():
            print(f"{k}: R2 = {v:.4f}")
