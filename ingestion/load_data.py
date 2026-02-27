# ingestion/load_data.py
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# ── STEP 1: Load credentials from .env file ──
load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# ── STEP 2: Create database connection ──
# SQLAlchemy creates a connection engine to PostgreSQL
# Think of this as opening a channel to your database
engine = create_engine(
    f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
)

def explore_data(df):
    """
    Before loading anything into the database we always explore first
    This tells us what we're working with — shape, columns, nulls, data types
    Professional analysts never skip this step
    """
    print("=" * 50)
    print("DATASET EXPLORATION")
    print("=" * 50)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumn names: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nNull values:\n{df.isnull().sum()}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
    print(f"\nFraud distribution:")
    print(df['isFraud'].value_counts())
    print(f"\nTransaction types:")
    print(df['type'].value_counts())
    print("=" * 50)

def load_raw_transactions():
    """
    Loads raw PaySim data into PostgreSQL
    We load 100,000 rows — enough to build everything without slowing down
    """

    print("Reading CSV file...")

    # Read first 100,000 rows
    # The full dataset has 6M rows — too slow for development
    # 100k rows is enough to build and demonstrate everything
    df = pd.read_csv(
        'data/PaySim.csv',
        nrows=100000
    )

    # Explore before loading
    explore_data(df)

    # ── STEP 3: Clean column names ──
    # PostgreSQL prefers lowercase column names with no spaces
    # This prevents errors when writing SQL queries later
    df.columns = df.columns.str.lower().str.strip()

    print("\nLoading into PostgreSQL...")
    print("This may take 30-60 seconds...")

    # ── STEP 4: Load into PostgreSQL ──
    # if_exists='replace' — drops and recreates table each time
    # index=False — don't write DataFrame row numbers as a column
    # chunksize=10000 — loads 10k rows at a time, more memory efficient
    df.to_sql(
        name='raw_transactions',
        con=engine,
        schema='public',
        if_exists='replace',
        index=False,
        chunksize=10000
    )

    print(f"\n✅ Successfully loaded {len(df):,} rows into raw_transactions table")
    print("Open pgAdmin and check — you should see the raw_transactions table")

    return df

if __name__ == '__main__':
    df = load_raw_transactions()