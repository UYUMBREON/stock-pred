import json
import sys
import os
import requests
import datetime
from pathlib import Path

import pandas as pd
from config import Config

pd.set_option("display.max_columns", None)

API_URL = "https://api.jquants.com"

# Load refresh token from environment variable
refreshtoken = os.getenv('JQUANTS_REFRESH_TOKEN')
if not refreshtoken:
    print("Error: JQUANTS_REFRESH_TOKEN environment variable not set!")
    print("Please create a .env file with your refresh token.")
    sys.exit(1)

# Load configuration
config = Config("/app/config.json")

# Create data directory if it doesn't exist
data_dir = Path(os.path.dirname(config.data.stock_list_path)) # /app/data
price_data_dir = Path(config.data.price_data_path) # /app/data/price_data/
data_dir.mkdir(exist_ok=True)
price_data_dir.mkdir(exist_ok=True) # Make sure price data dir is also created

# idToken取得
res = requests.post(f"{API_URL}/v1/token/auth_refresh?refreshtoken={refreshtoken}")
if res.status_code == 200:
    id_token = res.json()['idToken']
    headers = {'Authorization': 'Bearer {}'.format(id_token)}
    print("idTokenの取得に成功しました。")
else:
    print(res.json()["message"])
    sys.exit(1)

date = datetime.date.today().strftime('%Y%m%d')

params = {}
params["date"] = date

# Get listed companies info
print(f"Fetching listed companies info for date: {date}")
res = requests.get(f"{API_URL}/v1/listed/info", params=params, headers=headers)

if res.status_code == 200:
    d = res.json()
    data = d["info"]
    while "pagination_key" in d:
        params["pagination_key"] = d["pagination_key"]
        res = requests.get(f"{API_URL}/v1/listed/info", params=params, headers=headers)
        d = res.json()
        data += d["info"]
    
    list_df = pd.DataFrame(data)
    print(f"Found {len(list_df)} companies")
    
    # Save to data directory
    list_csv_path = Path(config.data.stock_list_path) # <-- Use config
    list_df.to_csv(list_csv_path, index=False)
    print(f"Stock list saved to: {list_csv_path}")
else:
    print("Error fetching listed companies:")
    print(res.json())
    sys.exit(1)

# Get daily quotes for each company
print("\nFetching daily quotes for each company...")
unique_codes = list_df["Code"].unique()

# Testing mode - only process first 5 companies
if False:  # Change to False when ready for full run
    unique_codes = unique_codes[:5]
    print(f"Testing mode: Processing only {len(unique_codes)} companies")

for i, code in enumerate(unique_codes, 1):
    print(f"Processing {code} ({i}/{len(unique_codes)})")
    
    params = {"code": code}
    
    res = requests.get(f"{API_URL}/v1/prices/daily_quotes", params=params, headers=headers)
    
    if res.status_code == 200:
        d = res.json()
        data = d["daily_quotes"]
        while "pagination_key" in d:
            params["pagination_key"] = d["pagination_key"]
            res = requests.get(f"{API_URL}/v1/prices/daily_quotes", params=params, headers=headers)
            d = res.json()
            data += d["daily_quotes"]
        
        df = pd.DataFrame(data)
        
        # Save to data directory with proper filename
        csv_path = price_data_dir / f"{code}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved {len(df)} records to: {csv_path}")
    else:
        print(f"  Error fetching data for {code}:")
        print(f"  {res.json()}")

print("\nData fetching completed!")

# --- Configuration ---
volume_column = "Volume"
threshold = 50000

# --- Processing ---
deleted_codes = []  # 削除されたコードを記録

for filename in os.listdir(price_data_dir):
    if not filename.lower().endswith(".csv"):
        continue  # skip non-csv files

    file_path = os.path.join(price_data_dir, filename)

    try:
        df = pd.read_csv(file_path)

        # Skip files without the target column
        if volume_column not in df.columns:
            print(f"Skipping {filename}: '{volume_column}' column not found.")
            continue

        median_volume = df[volume_column].median()

        if pd.isna(median_volume) or median_volume <= threshold:
            # コード（拡張子なし）を記録
            code = filename.replace('.csv', '')
            deleted_codes.append(code)
            
            os.remove(file_path)
            print(f"Deleted {filename}: median volume = {median_volume}")
        else:
            print(f"Kept {filename}: median volume = {median_volume}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# stock_list.csvから削除されたコードの行を除外
if deleted_codes:
    print(f"\nRemoving {len(deleted_codes)} codes from stock_list.csv...")
    list_csv_path = Path(config.data.stock_list_path)
    list_df = pd.read_csv(list_csv_path)
    original_count = len(list_df)
    
    list_df = list_df[~list_df["Code"].isin(deleted_codes)]

    median_volumes = {}
    for filename in os.listdir(price_data_dir):
        if not filename.lower().endswith(".csv"):
            continue
        code = filename.replace('.csv', '')
        if code in list_df["Code"].values:
            file_path = os.path.join(price_data_dir, filename)
            try:
                df = pd.read_csv(file_path)
                if volume_column in df.columns:
                    median_volumes[code] = df[volume_column].median()
            except Exception as e:
                print(f"Error reading {filename} for sorting: {e}")

    # Add median volume column and sort
    list_df['MedianVolume'] = list_df['Code'].map(median_volumes)
    list_df = list_df.sort_values('MedianVolume', ascending=False)
    list_df = list_df.drop('MedianVolume', axis=1)  # Remove helper column
    
    list_df.to_csv(list_csv_path, index=False)
    
    print(f"Updated stock_list.csv: {original_count} -> {len(list_df)} companies")
else:
    print("\nNo codes were deleted, stock_list.csv remains unchanged.")