# create SQL tables

# import necessary libraries
from sqlalchemy import create_engine, Column, Integer, String, inspect, ForeignKey, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import pandas as pd
from sqlalchemy.exc import IntegrityError
import os
import json
from datetime import datetime
import time

# specify folder path to all json files:
folder_path = "Data/all_json/"


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
    city = Column(String(100), nullable=True)

# Create a teams master table:
class Team(Base):
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_name = Column(String(150), nullable=False, unique=True)

# Create a matches table:
class Match(Base):
    __tablename__ = 'matches'

    game_id = Column(Integer, primary_key=True)
    game_date = Column(Date, nullable=False)

    player_of_the_match_id = Column(Integer, ForeignKey('players.id'))
    batting_first_team_id = Column(Integer, ForeignKey('teams.id'))
    bowling_first_team_id = Column(Integer, ForeignKey('teams.id'))
    toss_winner = Column(Integer, ForeignKey('teams.id'))
    toss_decision = Column(String(20), nullable=False)

    venue_id = Column(Integer, ForeignKey('venues.ground_id'))
    format = Column(Integer, ForeignKey('match_formats.id'))

    umpire_1 = Column(Integer, ForeignKey('officials.id'))
    umpire_2 = Column(Integer, ForeignKey('officials.id'))

    winning_team = Column(Integer, ForeignKey('teams.id'))
    innings_defeat = Column(String(3), nullable=False)  # "yes" or "no"

    win_by_runs = Column(Integer, nullable=True)
    win_by_wickets = Column(Integer, nullable=True)

# Create a playing_elevens table:
class PlayingEleven(Base):
    __tablename__ = 'playing_elevens'
    
    game_id = Column(Integer, ForeignKey('matches.game_id'), primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'), primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'), primary_key=True)
    
# Create ball_by_ball table:
class BallByBall(Base):
    __tablename__ = 'ball_by_ball'

    game_id = Column(Integer, ForeignKey('matches.game_id'), primary_key=True)
    inning = Column(Integer, primary_key=True)
    ball_number = Column(Integer, primary_key=True)

    batting_team = Column(Integer, ForeignKey('teams.id'))
    striker_id = Column(Integer, ForeignKey('players.id'))
    nonstriker_id = Column(Integer, ForeignKey('players.id'))
    bowler_id = Column(Integer, ForeignKey('players.id'))

    batter_runs = Column(Integer, nullable=False)
    extras_runs = Column(Integer, nullable=False)
    extras_type = Column(String(20), nullable=True)

    wicket = Column(Integer, nullable=False)  # 0 or 1
    wicket_type = Column(String(50), nullable=True)

    total_team_runs = Column(Integer, nullable=False)
    total_team_wickets = Column(Integer, nullable=False)



# creating the table if it doesn't exist
create_table_if_not_exists(engine, Player)
create_table_if_not_exists(engine, Official)
create_table_if_not_exists(engine, MatchFormat)
create_table_if_not_exists(engine, Venue)
create_table_if_not_exists(engine, Team)
create_table_if_not_exists(engine, Match)
create_table_if_not_exists(engine, PlayingEleven)
create_table_if_not_exists(engine, BallByBall)



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


# Create a session to input data into the teams table:            
Session = sessionmaker(bind=engine)
session = Session()

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # Only process JSON files
        # extract relative filepath
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # extract teams:
        teams = data.get("info", {}).get("teams", [])
        for team_name in teams:
            if not session.query(Team).filter_by(team_name=team_name).first():
                session.add(Team(team_name=team_name))
                session.commit()
                print(f"Added team: {team_name}")
            else:
                print(f"Team already exists: {team_name}")

# close session:
session.close()


# Create a session to input data into the matches table:
Session = sessionmaker(bind=engine)
session = Session()

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        try:
            game_id = int(filename.replace(".json", ""))
            filepath = os.path.join(folder_path, filename)

            with open(filepath, 'r') as f:
                data = json.load(f)

            info = data.get("info", {})
            registry = info.get("registry", {}).get("people", {})
            innings = data.get("innings", [])

            # Game date
            date_str = info.get("dates", [])[0]
            game_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            # Player of match
            pom_name = info.get("player_of_match", [None])[0]
            pom_id = None
            if pom_name:
                cricsheet_id = registry.get(pom_name)
                pom = session.query(Player).filter_by(cricsheet_id=cricsheet_id).first()
                pom_id = pom.id if pom else None

            # Teams
            teams = info.get("teams", [])
            team_name_1 = innings[0]['team']
            team_name_2 = [t for t in teams if t != team_name_1][0]

            team1 = session.query(Team).filter_by(team_name=team_name_1).first()
            team2 = session.query(Team).filter_by(team_name=team_name_2).first()
            batting_first_id = team1.id
            bowling_first_id = team2.id

            # Toss info
            toss_info = info.get("toss", {})
            toss_winner_name = toss_info.get("winner")
            toss_decision = toss_info.get("decision")
            toss_team = session.query(Team).filter_by(team_name=toss_winner_name).first()
            toss_winner_id = toss_team.id

            # Venue
            venue_name = info.get("venue")
            venue = session.query(Venue).filter_by(ground_name=venue_name).first()
            venue_id = venue.ground_id if venue else None

            # Match format
            format_name = info.get("match_type")
            match_format = session.query(MatchFormat).filter_by(match_format=format_name).first()
            format_id = match_format.id if match_format else None

            # Umpires
            umpires = info.get("officials", {}).get("umpires", [])
            umpire1_id, umpire2_id = None, None
            if len(umpires) >= 1:
                ump1 = session.query(Official).filter_by(umpire_name=umpires[0]).first()
                umpire1_id = ump1.id if ump1 else None
            if len(umpires) >= 2:
                ump2 = session.query(Official).filter_by(umpire_name=umpires[1]).first()
                umpire2_id = ump2.id if ump2 else None

            # Outcome
            outcome = info.get("outcome", {})
            winner_name = outcome.get("winner")
            winning_team_id = None
            if winner_name:
                winning_team = session.query(Team).filter_by(team_name=winner_name).first()
                winning_team_id = winning_team.id if winning_team else None

            # Innings defeat
            by = outcome.get("by", {})
            innings_defeat = "yes" if by.get("innings") == 1 else "no"

            # Win by runs/wickets
            win_by_runs = by.get("runs")
            win_by_wickets = by.get("wickets")

            # Check if match already exists
            exists = session.query(Match).filter_by(game_id=game_id).first()
            if not exists:
                match = Match(
                    game_id=game_id,
                    game_date=game_date,
                    player_of_the_match_id=pom_id,
                    batting_first_team_id=batting_first_id,
                    bowling_first_team_id=bowling_first_id,
                    toss_winner=toss_winner_id,
                    toss_decision=toss_decision,
                    venue_id=venue_id,
                    format=format_id,
                    umpire_1=umpire1_id,
                    umpire_2=umpire2_id,
                    winning_team=winning_team_id,
                    innings_defeat=innings_defeat,
                    win_by_runs=win_by_runs,
                    win_by_wickets=win_by_wickets
                )
                session.add(match)
                print(f"Match added: game_id={game_id}")
            else:
                print(f"Match already exists: game_id={game_id}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# close session
session.commit()
session.close()


# Create a session to input data into the playing elevens table:
Session = sessionmaker(bind=engine)
session = Session()

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        try:
            # Extract game_id from filename (as an integer)
            game_id = int(filename.replace(".json", ""))

            # Load the JSON file
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # extract player names and ids from the json file:
            info = data.get("info", {})
            players_by_team = info.get("players", {})
            registry = info.get("registry", {}).get("people", {})
            
            for team_name, player_names in players_by_team.items():
                # Check if team exists, and add if missing
                team = session.query(Team).filter_by(team_name=team_name).first()
                if not team:
                    team = Team(team_name=team_name)
                    session.add(team)
                    session.commit()
                    print(f"Added new team: {team_name}")
                team_id = team.id
                
                for player_name in player_names:
                    cricsheet_id = registry.get(player_name)
                    
                    # Check if player exists in the master table
                    player = session.query(Player).filter_by(cricsheet_id=cricsheet_id).first()
                    if not player:
                        # Add new player to master table
                        player = Player(
                            name=player_name,
                            unique_name=player_name,
                            cricsheet_id=cricsheet_id
                        )
                        session.add(player)
                        session.commit()
                        print(f"Added new player: {player_name} (cricsheet_id={cricsheet_id})")

                    player_id = player.id
                    
                    # Check for duplicate entry
                    exists = session.query(PlayingEleven).filter_by(
                        game_id=game_id,
                        player_id=player_id,
                        team_id=team_id
                    ).first()

                    if not exists:
                        entry = PlayingEleven(
                            game_id=game_id,
                            player_id=player_id,
                            team_id=team_id
                        )
                        session.add(entry)
                        print(f"Added: Game {game_id}, Team {team_name}, Player {player_name}")
                    else:
                        print(f"Already exists: Game {game_id}, Team {team_name}, Player {player_name}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Commit and close session
session.commit()
session.close()


# Create a session to input data into the ball by ball table:
Session = sessionmaker(bind=engine)
session = Session()
total_files = 1
# Loop through all JSON files
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        try:
            game_id = int(filename.replace(".json", ""))
            filepath = os.path.join(folder_path, filename)

            with open(filepath, 'r') as f:
                data = json.load(f)

            registry = data['info']['registry']['people']
            innings_data = data['innings']

            inning_num = 0

            for inning in innings_data:
                inning_num += 1
                team_name = inning['team']
                team = session.query(Team).filter_by(team_name=team_name).first()
                if not team:
                    print(f"⚠️ Team not found: {team_name}")
                    continue
                team_id = team.id

                ball_number = 0
                total_runs = 0
                total_wickets = 0

                for over in inning['overs']:
                    for delivery in over['deliveries']:
                        ball_number += 1

                        batter = delivery.get("batter")
                        bowler = delivery.get("bowler")
                        non_striker = delivery.get("non_striker")

                        batter_id = session.query(Player).filter_by(cricsheet_id=registry.get(batter)).first().id
                        bowler_id = session.query(Player).filter_by(cricsheet_id=registry.get(bowler)).first().id
                        nonstriker_id = session.query(Player).filter_by(cricsheet_id=registry.get(non_striker)).first().id

                        runs = delivery["runs"]
                        batter_runs = runs.get("batter", 0)
                        extras_runs = runs.get("extras", 0)

                        extras_type = None
                        if "extras" in delivery:
                            extras_type = list(delivery["extras"].keys())[0]

                        wicket_info = delivery.get("wickets", [])
                        wicket = 1 if wicket_info else 0
                        wicket_type = wicket_info[0]["kind"] if wicket_info else None

                        total_runs += batter_runs + extras_runs
                        total_wickets += wicket

                        # Insert row
                        ball = BallByBall(
                            game_id=game_id,
                            inning=inning_num,
                            ball_number=ball_number,
                            batting_team=team_id,
                            striker_id=batter_id,
                            nonstriker_id=nonstriker_id,
                            bowler_id=bowler_id,
                            batter_runs=batter_runs,
                            extras_runs=extras_runs,
                            extras_type=extras_type,
                            wicket=wicket,
                            wicket_type=wicket_type,
                            total_team_runs=total_runs,
                            total_team_wickets=total_wickets
                        )
                        session.add(ball)

            # ✅ Commit after processing one file
            session.commit()
            print(f"Committed: {filename}")
            print(total_files)
            total_files += 1
            time.sleep(1)

        except Exception as e:
            session.rollback()  # Roll back any partial changes for this file
            print(f"Error processing {filename}: {e}")

# Close session after all files
session.close()

# committed the first 2149 files...
