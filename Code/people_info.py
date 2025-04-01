# This file takes the people.csv data from the Data folder and adds it to the SQL databse 

# import necessary libraries:
from helper_functions import move_from_python_to_sql
import pandas as pd


# add people data to SQL database
people_data = pd.read_csv('Data/people.csv')
move_from_python_to_sql(people_data, 'People')


