#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from datetime import datetime
import pandas as pd
import requests
import sqlite3
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from entsoe import EntsoePandasClient
from dotenv import load_dotenv
from tqdm import tqdm

# --------------------------------------------------
# Configuration & Setup
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Set up requests session with retry logic
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504, 520, 522])
adapter = HTTPAdapter(max_retries=retries)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("ENTSOE_API_KEY")
if not API_KEY:
    logging.error("ENTSOE_API_KEY not found in .env file. Please set it and try again.")
    sys.exit(1)

# Initialize the ENTSO-E client with the custom session
client = EntsoePandasClient(api_key=API_KEY, session=session)

# Use Germany's bidding zone EIC code for both queries
BIDDING_ZONE = "10Y1001A1001A83F"

# Define neighboring countries and their EIC codes
NEIGHBORING_COUNTRIES = {
    'FR': '10YFR-RTE------C',  # France
    'NL': '10YNL----------L',  # Netherlands
    'BE': '10YBE----------2',  # Belgium
    'PL': '10YPL-AREA-----S',  # Poland
    'CZ': '10YCZ-CEPS-----N',  # Czech Republic
    'AT': '10YAT-APG------L',  # Austria
    'CH': '10YCH-SWISSGRIDZ',  # Switzerland
    'DK': '10Y1001A1001A65H'   # Denmark
}

# Define the overall time range for data requests:
start_overall = pd.Timestamp('2021-01-01 00:00', tz='Europe/Berlin')
end_overall   = pd.Timestamp('2025-01-01 00:00', tz='Europe/Berlin')  # end is exclusive

# Create the output directories if they don't exist
output_dir = os.path.join("data", "raw")
sql_dir = os.path.join("data", "sql")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(sql_dir, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Download electricity data from ENTSO-E')
    parser.add_argument('--generation', action='store_true', help='Download generation data')
    parser.add_argument('--load', action='store_true', help='Download load data')
    parser.add_argument('--flows', action='store_true', help='Download cross-border flows')
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    return parser.parse_args()

def get_rolling_ranges(start, end, window_days=7):
    """
    Given a start and end timestamp, generate rolling (start, end) tuples for window_days.
    """
    ranges = []
    current = start
    while current < end:
        next_time = current + pd.Timedelta(days=window_days)
        if next_time > end:
            next_time = end
        ranges.append((current, next_time))
        current = next_time
    return ranges

def fetch_rolling_data(fetch_func, zone, date_ranges, dataset_name):
    """
    For each rolling period, call fetch_func(zone, start, end) and return a concatenated DataFrame.
    If any call returns an empty DataFrame, exit immediately.
    """
    dfs = []
    for start, end in tqdm(date_ranges, desc=f"Downloading {dataset_name}", unit="window"):
        try:
            df = fetch_func(zone, start=start, end=end)
            if df is None or df.empty:
                logging.error(f"Empty data returned for {dataset_name} data for period {start} to {end}")
                return pd.DataFrame()
            df = df.copy()
            df.reset_index(inplace=True)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error fetching {dataset_name} data for period {start} to {end}: {e}")
            return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Add this new function after the existing fetch_rolling_data function

def fetch_flows_data(fetch_func, zone, date_ranges, dataset_name):
    """
    Special handling for cross-border flows which return Series instead of DataFrames.
    """
    dfs = []
    for start, end in tqdm(date_ranges, desc=f"Downloading {dataset_name}", unit="window"):
        try:
            df = fetch_func(zone, start=start, end=end)
            if df is None or df.empty:
                logging.warning(f"Empty data returned for {dataset_name} data for period {start} to {end}")
                continue
                
            # Handle Series return from cross-border flows
            if isinstance(df, pd.Series):
                df = df.to_frame(name='value')
                df = df.reset_index()
            
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error fetching {dataset_name} data for period {start} to {end}: {e}")
            continue
            
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def get_cross_border_flows(client, zone, date_ranges):
    """Get power exchanges with neighboring countries"""
    all_flows = []
    
    for country_code, country_zone in NEIGHBORING_COUNTRIES.items():
        logging.info(f"Fetching flows between Germany and {country_code}...")
        
        # Get flows in both directions
        try:
            # Germany to neighbor
            df_out = fetch_flows_data(
                lambda z, start, end: client.query_crossborder_flows(z, country_zone, start=start, end=end),
                zone, 
                date_ranges,
                f"flows_DE_to_{country_code}"
            )
            if not df_out.empty:
                df_out['border_from'] = 'DE'
                df_out['border_to'] = country_code
                all_flows.append(df_out)
            
            # Neighbor to Germany
            df_in = fetch_flows_data(
                lambda z, start, end: client.query_crossborder_flows(country_zone, z, start=start, end=end),
                zone,
                date_ranges,
                f"flows_{country_code}_to_DE"
            )
            if not df_in.empty:
                df_in['border_from'] = country_code
                df_in['border_to'] = 'DE'
                all_flows.append(df_in)
                
        except Exception as e:
            logging.error(f"Error fetching flows with {country_code}: {str(e)}")
            continue
    
    if all_flows:
        return pd.concat(all_flows, ignore_index=True)
    return pd.DataFrame()

def validate_data_alignment(generation_df, load_df):
    """Validate temporal alignment and completeness of both datasets"""
    load_times = set(load_df['datetime'])
    gen_times = set(generation_df['datetime'])
    
    missing_in_load = gen_times - load_times
    missing_in_gen = load_times - gen_times
    
    if missing_in_load or missing_in_gen:
        logging.warning(f"Missing timestamps in load: {len(missing_in_load)}")
        logging.warning(f"Missing timestamps in generation: {len(missing_in_gen)}")
    
    return len(missing_in_load) == 0 and len(missing_in_gen) == 0

def setup_sqlite_db(db_path):
    """Create SQLite database and tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generation (
            datetime TEXT,
            production_type TEXT,
            value FLOAT,
            PRIMARY KEY (datetime, production_type)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS load (
            datetime TEXT PRIMARY KEY,
            value FLOAT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cross_border_flows (
            datetime TEXT,
            border_from TEXT,
            border_to TEXT,
            value FLOAT,
            PRIMARY KEY (datetime, border_from, border_to)
        )
    ''')
    
    conn.commit()
    return conn

def save_to_sqlite(conn, df, table_name):
    """Save DataFrame to SQLite table"""
    if not df.empty:
        try:
            # Create a copy to avoid modifying the original dataframe
            df_sql = df.copy()
            
            # Convert timestamp columns to string format
            datetime_cols = df_sql.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                df_sql[col] = df_sql[col].astype(str)
            
            # Handle specific table requirements
            if table_name == "cross_border_flows":
                if 'start' in df_sql.columns:
                    df_sql = df_sql.rename(columns={'start': 'datetime'})
                required_cols = ['datetime', 'border_from', 'border_to', 'value']
                df_sql = df_sql[required_cols]
            
            elif table_name == "generation":
                if 'production_type' not in df_sql.columns:
                    df_sql['production_type'] = 'total'
            
            elif table_name == "load":
                if 'value' not in df_sql.columns and 'load' in df_sql.columns:
                    df_sql = df_sql.rename(columns={'load': 'value'})
            
            df_sql.to_sql(table_name, conn, if_exists='replace', index=False)
            logging.info(f"Saved {len(df_sql)} rows to {table_name} table")
            
        except Exception as e:
            logging.error(f"Error saving to SQLite: {str(e)}")
            logging.error(f"Data saved to parquet file but not to SQLite for {table_name}")
    else:
        logging.warning(f"No data to save to {table_name} table")

def main():
    args = parse_args()
    if not (args.generation or args.load or args.flows or args.all):
        logging.error("Please specify which data to download (--generation, --load, --flows, or --all)")
        sys.exit(1)

    # Generate rolling ranges (7-day windows) from January 2021 to January 2025
    rolling_ranges = get_rolling_ranges(start_overall, end_overall, window_days=7)
    logging.info(f"Total windows to fetch: {len(rolling_ranges)}")
    
    # Set up SQLite database
    db_path = os.path.join(sql_dir, "electricity_data.db")
    conn = setup_sqlite_db(db_path)
    
    try:
        # Fetch generation data
        if args.generation or args.all:
            logging.info("Starting download for generation data...")
            generation_df = fetch_rolling_data(client.query_generation, BIDDING_ZONE, rolling_ranges, "generation")
            if not generation_df.empty:
                gen_file = os.path.join(output_dir, "generation.parquet")
                generation_df.to_parquet(gen_file, index=False)
                save_to_sqlite(conn, generation_df, "generation")
                logging.info(f"Saved generation data to {gen_file} and SQLite")
        
        # Fetch load data
        if args.load or args.all:
            logging.info("Starting download for load (demand) data...")
            load_df = fetch_rolling_data(client.query_load, BIDDING_ZONE, rolling_ranges, "load")
            if not load_df.empty:
                load_file = os.path.join(output_dir, "load.parquet")
                load_df.to_parquet(load_file, index=False)
                save_to_sqlite(conn, load_df, "load")
                logging.info(f"Saved load data to {load_file} and SQLite")
        
        # Fetch cross-border flows
        if args.flows or args.all:
            logging.info("Starting download for cross-border flows...")
            flows_df = get_cross_border_flows(client, BIDDING_ZONE, rolling_ranges)
            if not flows_df.empty:
                flows_file = os.path.join(output_dir, "cross_border_flows.parquet")
                flows_df.to_parquet(flows_file, index=False)
                save_to_sqlite(conn, flows_df, "cross_border_flows")
                logging.info(f"Saved {len(flows_df)} cross-border flow records")
            else:
                logging.warning("No cross-border flow data was collected")
        
        # Only validate if both datasets were downloaded
        if (args.generation or args.all) and (args.load or args.all):
            if 'generation_df' in locals() and 'load_df' in locals():
                if not generation_df.empty and not load_df.empty:
                    is_aligned = validate_data_alignment(generation_df, load_df)
                    logging.info(f"Data alignment validation {'passed' if is_aligned else 'failed'}")
    
    finally:
        # Ensure database connection is closed
        conn.close()
        logging.info("Data collection completed")

if __name__ == "__main__":
    main()