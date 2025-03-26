# Helper functions for rest of code

# Import necessary libraries:
from sqlalchemy import create_engine



# Helper function to move pandas dataframe from python to mysql database:
# Parameters:
#   data: pandas dataframe that follows the structure of the sql_table you intend to append it to
#   sql_table_name: name of the SQL table where you want to move your pandas df to
#   if_exists: can take 3 values --> 'append' (default), 'fail' (raises an error if table alr exists), 'replace' (drops table and creates a new one)
# Output:
#   data seen in mySQL database
def move_from_python_to_sql(data, sql_table_name, if_exists='append'):
    connection_string = "mysql+pymysql://root:root@localhost:8889/CricketData"
    engine = create_engine(connection_string)
    # Insert rows into the database
    try:
        data.to_sql(sql_table_name, con=engine, if_exists=if_exists, index=False)
        print("Rows successfully inserted into the database!")
    except Exception as e:
        print(f"Error inserting rows: {e}")