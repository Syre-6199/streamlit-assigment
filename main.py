# Comprehensive data quality check
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Basic dataset information
print("Dataset Shape:", X.shape)
print("\nFeature Names:")
for col in X.columns:
    print(f"- {col}")

# Missing values analysis
missing_values = pd.DataFrame({
    'Missing Count': X.isnull().sum(),
    'Missing Percentage': (X.isnull().sum() / len(X) * 100).round(2)
})
missing_values = missing_values.sort_values('Missing Count', ascending=False)

print("\nMissing Values Summary:")
print(missing_values[missing_values['Missing Count'] > 0])

if missing_values['Missing Count'].sum() == 0:
    print("\nNo missing values found in the dataset!")

# Basic statistics for each feature
print("\nBasic Statistics:")
print(X.describe().round(2))

# Visualize distribution of missing values (if any)
if missing_values['Missing Count'].sum() > 0:
    plt.figure(figsize=(12, 6))
    sns.heatmap(X.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.xlabel('Features')
    plt.show()

# Check for potential duplicates
duplicates = X.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Display first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(X.head())
