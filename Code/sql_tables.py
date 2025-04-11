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
from dotenv import load_dotenv

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



# For mySQL db hosted on AWS
# user: admin
# password: Papasdog123$
# databse name: cricket-test-database

# SQL connection details:
username = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_USER")
password = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_PW")
host = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_HOST")
port = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_PORT")
database = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_DBNAME")

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


# Create a session to input data into players, matches, and playing_elevens tables:
Session = sessionmaker(bind=engine)
session = Session()

commit_counter = 0
BATCH_SIZE = 5000

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        try:
            game_id = int(filename.replace(".json", ""))
            filepath = os.path.join(folder_path, filename)

            with open(filepath, 'r') as f:
                data = json.load(f)

            info = data.get("info", {})
            registry = info.get("registry", {}).get("people", {})
            players_by_team = info.get("players", {})
            innings = data.get("innings", [])

            # Step 1: Insert players
            cricsheet_ids_seen = set()
            for player_names in players_by_team.values():
                for player_name in player_names:
                    cricsheet_id = registry.get(player_name)
                    if not cricsheet_id or cricsheet_id in cricsheet_ids_seen:
                        continue
                    cricsheet_ids_seen.add(cricsheet_id)

                    player = session.query(Player).filter_by(cricsheet_id=cricsheet_id).first()
                    if not player:
                        player = Player(
                            name=player_name,
                            unique_name=player_name,
                            cricsheet_id=cricsheet_id
                        )
                        session.add(player)
                        session.flush()
                        print(f"🟢 Added player: {player_name} ({cricsheet_id})")

            # Step 2: Insert match
            if not session.query(Match).filter_by(game_id=game_id).first():
                game_date = datetime.strptime(info["dates"][0], "%Y-%m-%d").date()

                # Player of match
                pom_name = info.get("player_of_match", [None])[0]
                pom_id = None
                if pom_name and pom_name in registry:
                    pom = session.query(Player).filter_by(cricsheet_id=registry[pom_name]).first()
                    pom_id = pom.id if pom else None

                # Toss info
                toss = info.get("toss", {})
                toss_team = session.query(Team).filter_by(team_name=toss.get("winner")).first()
                toss_id = toss_team.id if toss_team else None
                toss_decision = toss.get("decision", "bat")

                # Batting/bowling order
                team1 = innings[0]['team']
                team2 = [t for t in info.get("teams", []) if t != team1][0]
                team1_id = session.query(Team).filter_by(team_name=team1).first().id
                team2_id = session.query(Team).filter_by(team_name=team2).first().id

                # Venue
                venue_name = info.get("venue")
                venue = session.query(Venue).filter_by(ground_name=venue_name).first()
                venue_id = venue.ground_id if venue else None

                # Format
                match_format = info.get("match_type")
                format_obj = session.query(MatchFormat).filter_by(match_format=match_format).first()
                format_id = format_obj.id if format_obj else None

                # Umpires
                umpires = info.get("officials", {}).get("umpires", [])
                ump1 = session.query(Official).filter_by(umpire_name=umpires[0]).first() if len(umpires) > 0 else None
                ump2 = session.query(Official).filter_by(umpire_name=umpires[1]).first() if len(umpires) > 1 else None

                # Outcome
                outcome = info.get("outcome", {})
                winner_team = session.query(Team).filter_by(team_name=outcome.get("winner")).first()
                winner_id = winner_team.id if winner_team else None

                innings_defeat = "yes" if outcome.get("by", {}).get("innings") == 1 else "no"
                win_by_runs = outcome.get("by", {}).get("runs")
                win_by_wickets = outcome.get("by", {}).get("wickets")

                session.add(Match(
                    game_id=game_id,
                    game_date=game_date,
                    player_of_the_match_id=pom_id,
                    batting_first_team_id=team1_id,
                    bowling_first_team_id=team2_id,
                    toss_winner=toss_id,
                    toss_decision=toss_decision,
                    venue_id=venue_id,
                    format=format_id,
                    umpire_1=ump1.id if ump1 else None,
                    umpire_2=ump2.id if ump2 else None,
                    winning_team=winner_id,
                    innings_defeat=innings_defeat,
                    win_by_runs=win_by_runs,
                    win_by_wickets=win_by_wickets
                ))
                print(f"🟣 Added match: {game_id}")
                commit_counter += 1

            # Step 3: Insert playing elevens
            for team_name, player_names in players_by_team.items():
                team = session.query(Team).filter_by(team_name=team_name).first()
                if not team:
                    team = Team(team_name=team_name)
                    session.add(team)
                    session.flush()
                team_id = team.id

                for player_name in player_names:
                    cricsheet_id = registry.get(player_name)
                    if not cricsheet_id:
                        continue

                    player = session.query(Player).filter_by(cricsheet_id=cricsheet_id).first()
                    if not player:
                        continue  # Should not happen if step 1 was successful

                    exists = session.query(PlayingEleven).filter_by(
                        game_id=game_id,
                        player_id=player.id,
                        team_id=team_id
                    ).first()
                    if not exists:
                        session.add(PlayingEleven(
                            game_id=game_id,
                            player_id=player.id,
                            team_id=team_id
                        ))
                        commit_counter += 1

            # Commit in batches
            if commit_counter >= BATCH_SIZE:
                session.commit()
                print(f"✅ Committed {commit_counter} rows.")
                commit_counter = 0

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            session.rollback()

# Final commit
if commit_counter > 0:
    session.commit()
    print(f"✅ Final commit of {commit_counter} remaining entries.")

session.close()



# Create a session to input data into the ball_by_ball table:
Session = sessionmaker(bind=engine)
session = Session()

# Cache player ID map
player_id_map = {p.cricsheet_id: p.id for p in session.query(Player).all()}
team_id_map = {t.team_name: t.id for t in session.query(Team).all()}
match_game_ids = set([m.game_id for m in session.query(Match.game_id).all()])

total_files = 1

for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        try:
            game_id = int(filename.replace(".json", ""))

            # Skip if match doesn't exist (foreign key constraint)
            if game_id not in match_game_ids:
                print(f"❌ Match not found for game_id={game_id}, skipping.")
                continue

            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

            registry = data['info']['registry']['people']
            innings_data = data['innings']

            balls_to_add = []

            for inning_num, inning in enumerate(innings_data, start=1):
                team_name = inning['team']
                team_id = team_id_map.get(team_name)
                if not team_id:
                    print(f"⚠️ Team not found: {team_name}")
                    continue

                ball_number = 0
                total_runs = 0
                total_wickets = 0

                for over in inning['overs']:
                    for delivery in over['deliveries']:
                        ball_number += 1

                        batter = registry.get(delivery.get("batter"))
                        bowler = registry.get(delivery.get("bowler"))
                        non_striker = registry.get(delivery.get("non_striker"))

                        batter_id = player_id_map.get(batter)
                        bowler_id = player_id_map.get(bowler)
                        nonstriker_id = player_id_map.get(non_striker)

                        if not all([batter_id, bowler_id, nonstriker_id]):
                            print(f"⚠️ Missing player ID for one or more participants, skipping delivery.")
                            continue

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

                        balls_to_add.append(BallByBall(
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
                        ))

            # Batch insert
            session.bulk_save_objects(balls_to_add)
            session.commit()
            print(f"✅ Committed: {filename} ({total_files})")
            total_files += 1

        except Exception as e:
            session.rollback()
            print(f"❌ Error processing {filename}: {e}")

# Close session after all files
session.close()