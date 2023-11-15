from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import os
import MySQLdb
load_dotenv()

engine = create_engine(
        f"mysql+mysqldb://{os.getenv('DATABASE_USERNAME')}:" \
        f"{os.getenv('DATABASE_PASSWORD')}@{os.getenv('DATABASE_HOST')}/" \
        f"{os.getenv('DATABASE')}?ssl_ca=ca-certificates.crt&ssl_mode=VERIFY_IDENTITY"
)

connection = MySQLdb.connect(
  host=os.getenv("DATABASE_HOST"),
  user=os.getenv("DATABASE_USERNAME"),
  passwd=os.getenv("DATABASE_PASSWORD"),
  db=os.getenv("DATABASE"),
  autocommit=True,
  ssl_mode="VERIFY_IDENTITY",
  ssl={ "ca": "ca-certificates.crt" }
)

# Function to write dataframe to SQL
def insert_dataframe_to_sql(table_name, dataframe, cursor):
    # Prepare the SQL insert statement
    placeholders = ', '.join(['%s'] * len(dataframe.columns))
    columns = ', '.join(dataframe.columns)

    # Prepare the ON DUPLICATE KEY UPDATE part of the query
    update_columns = ', '.join([f"{col} = VALUES({col})" for col in dataframe.columns])

    sql = f"""INSERT INTO {table_name} ({columns}) VALUES ({placeholders})
              ON DUPLICATE KEY UPDATE {update_columns}"""

    # Convert dataframe to a list of tuples, handling NaN values
    data = [tuple(row) if not any(pd.isna(val) for val in row) 
            else tuple(None if pd.isna(val) else val for val in row) 
            for row in dataframe.values]

    # Execute the SQL command for each row
    cursor.executemany(sql, data)