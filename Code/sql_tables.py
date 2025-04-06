# create SQL tables

# import necessary libraries
from sqlalchemy import create_engine, Column, Integer, String, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from sqlalchemy.exc import IntegrityError
import os
import json


# define create_table_if_not_exists function:
def create_table_if_not_exists(engine, model_class):
    inspector = inspect(engine)
    if not inspector.has_table(model_class.__tablename__):
        print(f"Creating table: {model_class.__tablename__}")
        model_class.__table__.create(engine)
    else:
        print(f"Table '{model_class.__tablename__}' already exists.")


# SQL connection details:
username = 'root'
password = 'root'
host = 'localhost'
port = '8889'
database = 'CricketData'

# create engine:
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}', echo=True)

# Base class
Base = declarative_base()


# Create a players master table:
class Player(Base):
    __tablename__ = 'players'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    unique_name = Column(String(100), nullable=False)
    cricsheet_id = Column(String(100), nullable=False, unique=True)

# Create an officials master table:
class Official(Base):
    __tablename__ = 'officials'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    umpire_name = Column(String(100), nullable=False, unique=True)

# Create a match format master table:
class MatchFormat(Base):
    __tablename__ = 'match_formats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_format = Column(String(50), nullable=False, unique=True)

# Create a venues master table:
class Venue(Base):
    __tablename__ = 'venues'
    
    ground_id = Column(Integer, primary_key=True, autoincrement=True)
    ground_name = Column(String(150), nullable=False, unique=True)
    city = Column(String(100), nullable=False)

# Create a teams master table: (still need to do this, do it later, leave blank for now)
#   
#
#
#

# creating the table if it doesn't exist
create_table_if_not_exists(engine, Player)
create_table_if_not_exists(engine, Official)
create_table_if_not_exists(engine, MatchFormat)
create_table_if_not_exists(engine, Venue)



# add data to players table:
df = pd.read_csv('Data/people.csv')



# Create a session to input player information in the players master table:
Session = sessionmaker(bind=engine)
session = Session()

# Loop through the dataframe and add players
for index, row in df.iterrows():
    player = Player(
        name=row['name'],  
        unique_name=row['unique_name'],
        cricsheet_id=row['identifier']  
    )
    session.add(player)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()  # clear the failed transaction if row is invalid
        print(f"Row already exists: cricsheet_id='{row['identifier']}'")

# close session
session.close()


# Create a session to input data into the match_format table:
Session = sessionmaker(bind=engine)
session = Session()

# specify folder path:
folder_path = "Data/all_json/"

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # Only process JSON files
        # extract relative filepath
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        # extract match format
        match_type = data["info"]["match_type"]
        
        # Check if match_format already exists
        exists = session.query(MatchFormat).filter_by(match_format=match_type).first()
        if not exists:
            game_format = MatchFormat(match_format=match_type)
            session.add(game_format)
            session.commit()
            print(f"Added: match_format='{match_type}'")
        else:
            print(f"Already exists: match_format='{match_type}'")

# close session:
session.close()


# Create a session to input data into the officials table:
Session = sessionmaker(bind=engine)
session = Session()

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # Only process JSON files
        # extract relative filepath
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Extract all types of officials
        officials = data.get("info", {}).get("officials", {})
        all_officials = []

        # Get all names from each official category
        all_officials.extend(officials.get("umpires", []))
        all_officials.extend(officials.get("tv_umpires", []))
        all_officials.extend(officials.get("match_referees", []))
        
        # Insert each official if not already in table
        for name in all_officials:
            if name:  # Skip empty strings
                exists = session.query(Official).filter_by(umpire_name=name).first()
                if not exists:
                    session.add(Official(umpire_name=name))
                    session.commit()
                    print(f"Added official: {name}")
                else:
                    print(f"Official already exists: {name}")

# close session:
session.close()


# Create a session to input data into the venues table:            
Session = sessionmaker(bind=engine)
session = Session()

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # Only process JSON files
        # extract relative filepath
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # extract venue and city data:
        info = data.get("info", {})
        city_name = info.get("city", "")
        venue = info.get("venue")
        
        # check if venue already exists:
        if venue:
            exists = session.query(Venue).filter_by(ground_name=venue).first()
            if not exists:
                ground = Venue(
                    ground_name = venue,
                    city = city_name
                )
                session.add(ground)
                session.commit()
                print(f"Added: ground_name='{venue}'")
            else:
                print(f"Already exists: ground_name='{venue}'")

# close session:
session.close()