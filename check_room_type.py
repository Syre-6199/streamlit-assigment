import pandas as pd
CSV = 'Airbnb_site_hotel new.csv'
try:
    df = pd.read_csv(CSV, low_memory=False)
    if 'room_type' in df.columns:
        print('Unique room_type values:')
        print(df['room_type'].unique())
        print('\nCounts:')
        print(df['room_type'].value_counts(dropna=False).head(50))
    else:
        print('No column named room_type found. Columns are:')
        print(df.columns.tolist())
except Exception as e:
    print('Error reading CSV:', e)
