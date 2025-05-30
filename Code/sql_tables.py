# Create SQL database table structures and populate tables with data

from sqlalchemy import create_engine, Column, Integer, String, inspect, ForeignKey, Date, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.exc import IntegrityError, SQLAlchemyError, OperationalError
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import re


Base = declarative_base()
# Define all models (Player, Official, MatchFormat, Venue, Team, Match, PlayingEleven, BallByBall)
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

    game_id = Column(String(25), primary_key=True)
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
    
    game_id = Column(String(25), ForeignKey('matches.game_id'), primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'), primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'), primary_key=True)
    
# Create ball_by_ball table:
class BallByBall(Base):
    __tablename__ = 'ball_by_ball'

    game_id = Column(String(25), ForeignKey('matches.game_id'), primary_key=True)
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

engine = 0
Session = 0
# define folder path to all .json data files
folder_path = "Data/all_json/"


# Load environment variables
def initialize_db_params():
    print("Initialize params")
    global engine
    global Session
    load_dotenv()
    username = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_USER")
    password = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_PW")
    host = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_HOST")
    port = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_PORT")
    database = os.getenv("UCMAS_AWS_CRIC01_DB_ADMIN_DBNAME")
    
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}', echo=False)
    Session = sessionmaker(bind=engine)



# Create tables if they do not exist, else continue:
# return True if function works, else return False
def create_table_if_not_exists(engine, model_class):
    print("Create table if it doesnt exist")
    try:
        inspector = inspect(engine)
        # Try a method that connects to the DB
        tables = inspector.get_table_names()
        print("✅ Tables in DB:", tables)
        
        if not inspector.has_table(model_class.__tablename__):
            print(f"Creating table: {model_class.__tablename__}")
            model_class.__table__.create(engine)
        else:
            print(f"Table '{model_class.__tablename__}' already exists. Skipping creation.")
        return True   # success
    except OperationalError as e:
        print("❌ OperationalError: Could not connect to the database.")
        print(e)
    except SQLAlchemyError as e:
        print("❌ SQLAlchemyError: Error inspecting the database.")
        print(e)
    except Exception as e:
        print("❌ Unexpected error: perhaps database credentials are invalid...")
        print(e)
    return False  # failure
    


def create_all_tables():
    print("Create all the tables")
    for model in [Player, Official, MatchFormat, Venue, Team, Match, PlayingEleven, BallByBall]:
        create_table_if_not_exists(engine, model)


# returns a list of all game_ids based on the .json files in the folder path:
def load_all_game_ids():
    print("Loading all files...")
    global folder_path
    list_of_all_ids = set()    # use set to avoid duplicates
    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue
        
        # Use full name before `.json` as the game ID
        game_id = os.path.splitext(filename)[0]
        
        if game_id in list_of_all_ids:
            raise ValueError(f"❌ Duplicate game ID found: {game_id} from file {filename}")
        else:
            list_of_all_ids.add(game_id)
            
    print(f"✅ Loaded {len(list_of_all_ids)} unique game IDs.")
    return list_of_all_ids
    
# add data to master tables (match formats, venues, teams, and officials):
def add_matchformats_venues_teams_officials(session, game_ids):
    # Preload existing DB entries
    # creates a set, avoiding duplicates
    #   Ex: if the row is <MatchFormat(match_format='T20')>, then f.match_format is 'T20'
    existing_formats = {f.match_format for f in session.query(MatchFormat).all()}
    existing_officials = {o.umpire_name for o in session.query(Official).all()}
    existing_venues = {v.ground_name for v in session.query(Venue).all()}
    existing_teams = {t.team_name for t in session.query(Team).all()}
    
    # Batch insert containers (lists that will store new objects to be inserted into the database)
    # This is quicker than inserting one row at a time
    formats_to_add, officials_to_add, venues_to_add, teams_to_add = [], [], [], []
    
    # loop thru list of all game_ids:
    for game_id in game_ids:
        # get filename and relative filepath:
        filename = f"{game_id}.json"
        filepath = os.path.join(folder_path, filename)
        
        # raise error if file is not found:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ File not found for game_id '{game_id}': {filename}")
        
        # open the file and load the data
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # extract appropriate data   
        info = data.get("info", {})
        match_type = info.get("match_type")
        venue = info.get("venue")
        city = info.get("city", "")
        teams = info.get("teams", [])
        officials = info.get("officials", {})
        
        # Match Format:
        # .append prepares the match type for database insertion
        # .add adds it to the set of already seen match formats to ensure no duplicates
        if match_type and match_type not in existing_formats:
            print(f"🆕 Adding match format: {match_type}")
            formats_to_add.append(MatchFormat(match_format=match_type))
            existing_formats.add(match_type)
        
        # Venue:
        if venue and venue not in existing_venues:
            print(f"🆕 Adding venue: {venue} (City: {city})")
            venues_to_add.append(Venue(ground_name=venue, city=city))
            existing_venues.add(venue)
        
        # Teams:
        # this one has a for loop because there are multiple teams for a game
        for team in teams:
            if team not in existing_teams:
                print(f"🆕 Adding team: {team}")
                teams_to_add.append(Team(team_name=team))
                existing_teams.add(team)
        
        # Officials:
        # officials include umpires, tv_umpires, and match_referees:
        # for each category, there can be multiple names (especially the umpires category, there are always 2 umpires)
        for category in ["umpires", "tv_umpires", "match_referees"]:
            for name in officials.get(category, []):
                if name and name not in existing_officials:
                    print(f"🆕 Adding official: {name} ({category})")
                    officials_to_add.append(Official(umpire_name=name))
                    existing_officials.add(name)
        
        # bulk_save objects performs a bulk insert of all the objects into the database in one efficient query
        # the statements only get executed if there is something that needs to be added
        # FORMATS:
        if formats_to_add:
            session.bulk_save_objects(formats_to_add)
            print(f"Inserted {len(formats_to_add)} match formats")
        # VENUES:
        if venues_to_add:
            session.bulk_save_objects(venues_to_add)
            print(f"Inserted {len(venues_to_add)} venues")     
        # TEAMS:
        if teams_to_add:
            session.bulk_save_objects(teams_to_add)
            print(f"Inserted {len(teams_to_add)} teams")
        # OFFICIALS:   
        if officials_to_add:
            session.bulk_save_objects(officials_to_add)
            print(f"Inserted {len(officials_to_add)} officials")
        
    # commit data additions:
    session.commit()
    # print done statement:        
    print("✅ Done adding data to match_formats, venues, officials, and teams tables!!!")


# add data to more tables (players, matches, playing_elevens):
def add_players_matches_playingelevens(session, game_ids):
    # Preload existing DB entries
    # creates a set, avoiding duplicates
    #   Ex: if the row is <MatchFormat(match_format='T20')>, then f.match_format is 'T20'
    existing_players = {p.cricsheet_id: p.id for p in session.query(Player).all()}
    existing_matches = {m.game_id for m in session.query(Match).with_entities(Match.game_id)}
    existing_playing_elevens = {
        (pe.game_id, pe.player_id, pe.team_id)
        for pe in session.query(PlayingEleven).all()
    }
    
    # Create maps:
    # Ex: teams map queries all teams from the teams table:
    #   then for each team object, extracts the team name as well as the ID associated with it
    #   helps translate names into IDs on the fly 
    teams_map = {t.team_name: t.id for t in session.query(Team).all()}
    formats_map = {f.match_format: f.id for f in session.query(MatchFormat).all()}
    officials_map = {o.umpire_name: o.id for o in session.query(Official).all()}
    venues_map = {v.ground_name: v.ground_id for v in session.query(Venue).all()}

    # Batch insert containers (lists that will store new objects to be inserted into the database)
    # This is quicker than inserting one row at a time
    matches_to_add = []
    players_to_add = []
    playing_elevens_to_add = []
    total_counter = 1
    # loop thru list of all game ids:
    for game_id in game_ids:
        # get the filename and the relative filepath
        filename = f"{game_id}.json"
        filepath = os.path.join(folder_path, filename)
        
        # Raise error if file is not found and exit:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ File not found: {filename}")
    
        # Load the json file's data:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # extract appropriate data:
        info = data.get("info", {})
        registry = info.get("registry", {}).get("people", {})
        players_section = info.get("players", {})
        
        # PLAYERS:
        # add any missing players based on cricsheet id as seen in the registry:
        # loop through every entry in the registry['people'] dictionary:
        for name, cricsheet_id in registry.items():
            # id the cricsheet id does not exist in the database:
            if cricsheet_id not in existing_players:
                # create a new player object
                player = Player(name=name, unique_name=name, cricsheet_id=cricsheet_id)
                # adds player object to list for bulk insertion later:
                players_to_add.append(player)
                # add temporary placeholder ID, to say that we have handled this cricsheet_id during this run, 
                #   though the actual ID hasn't been auto-generated yet since it hasn't been committed
                existing_players[cricsheet_id] = None
                print(f"🆕 Added player: {name} with ID: {cricsheet_id}")
        
        # MATCH:
        # if the game id is not already in the matches table...
        if game_id not in existing_matches:
            # flush new players to be added in case POM is a new player that is not yet in the players table:
            if players_to_add:
                print(f"✅ Inserting {len(players_to_add)} players...")
                session.bulk_save_objects(players_to_add)
                session.flush()
                players_to_add.clear()
                existing_players = {p.cricsheet_id: p.id for p in session.query(Player).all()}
            # extract the necessary fields:
            venue_id = venues_map.get(info.get("venue"))
            format_id = formats_map.get(info.get("match_type"))
            toss_winner = teams_map.get(info.get("toss", {}).get("winner"))
            toss_decision = info.get("toss", {}).get("decision")
            # get both umpire IDs based on where the official name matches in the officials map 
            umpire_names = info.get("officials", {}).get("umpires", [])
            ump1 = officials_map.get(umpire_names[0]) if len(umpire_names) > 0 else None
            ump2 = officials_map.get(umpire_names[1]) if len(umpire_names) > 1 else None
            if len(umpire_names) < 2:
                print(f"⚠️ Only {len(umpire_names)} umpire(s) listed in game {game_id}")
            # get the first day of the game as the game date, and ensure it is stored as a date type:
            game_date = datetime.strptime(info["dates"][0], "%Y-%m-%d").date()
            winning_team = teams_map.get(info.get("outcome", {}).get("winner"))
            # victory margin:
            win_by = info.get("outcome", {}).get("by", {})
            win_by_runs = win_by.get("runs")  # could be None
            win_by_wickets = win_by.get("wickets")  # could be None
            # innings defeat:
            innings_defeat = "yes" if "innings" in win_by else "no"
            # Batting and bowling first teams
            teams = info.get("teams", [])
            batting_first_team = None
            innings = data.get("innings", [])
            if innings:
                batting_first_team = innings[0].get("team")
            bowling_first_team = None
            if batting_first_team:
                for team in teams:
                    if team != batting_first_team:
                        bowling_first_team = team
                        break  # Found the other team, stop looping
            batting_first_team_id = teams_map.get(batting_first_team)
            bowling_first_team_id = teams_map.get(bowling_first_team)
            # player of the match:
            player_of_the_match_id = None
            pom_names = info.get("player_of_match", [])
            if pom_names:
                pom_name = pom_names[0]  # Cricsheet always lists a single name
                cricsheet_id = registry.get(pom_name)
                if cricsheet_id:
                    player_of_the_match_id = existing_players.get(cricsheet_id)
                    # if POM ID is invalid or name is not found...
                    if player_of_the_match_id is None:
                        print(f"⚠️ Player of the match '{pom_name}' with Cricsheet ID '{cricsheet_id}' not found in DB for game {game_id}")
                else:
                    print(f"⚠️ Cricsheet ID not found for player of the match '{pom_name}' in game {game_id}")
            
            # save match object as "match"        
            match = Match(
                game_id=game_id,
                game_date=game_date,
                toss_winner=toss_winner,
                toss_decision=toss_decision,
                umpire_1=ump1,
                umpire_2=ump2,
                venue_id=venue_id,
                format=format_id,
                innings_defeat=innings_defeat,
                winning_team=winning_team,
                win_by_runs=win_by_runs,
                win_by_wickets=win_by_wickets,
                batting_first_team_id=batting_first_team_id,
                bowling_first_team_id=bowling_first_team_id,
                player_of_the_match_id=player_of_the_match_id
            )
            
            # add match object:
            # .append prepares the match for database insertion
            # .add adds it to the set of already seen matches to ensure no duplicates
            matches_to_add.append(match)
            session.bulk_save_objects(matches_to_add)
            session.flush()
            session.commit()
            matches_to_add.clear()
            existing_matches.add(game_id)
            print(f"🆕 Added and committed match: {game_id}")
        
        # PLAYING ELEVENS:
        # for each player name for each team in the players section:
        for team_name, player_names in players_section.items():
            team_id = teams_map.get(team_name)
            # raise a value error if you find an unknown team:
            if not team_id:
                raise ValueError(f"❌ Unknown team: {team_name} in game {game_id}")
            for player_name in player_names:
                cricsheet_id = registry.get(player_name)
                # raise a value error if the cricsheet ID cannot be found for that player name in the registry:
                if not cricsheet_id:
                    raise ValueError(f"❌ Cricsheet ID not found for player: {player_name} in game {game_id}")
                
                player_id = existing_players.get(cricsheet_id)
                if player_id is None:
                    # Player just added, flush to get ID
                    session.bulk_save_objects(players_to_add)
                    session.flush()
                    players_to_add.clear()
                    existing_players = {p.cricsheet_id: p.id for p in session.query(Player).all()}
                    player_id = existing_players[cricsheet_id]
                    
                key = (game_id, player_id, team_id)
                if key not in existing_playing_elevens:
                    playing_elevens_to_add.append(PlayingEleven(
                        game_id=game_id,
                        player_id=player_id,
                        team_id=team_id
                    ))
                    existing_playing_elevens.add(key)
                    print(f"🆕 Added Playing XI: {player_name} ({team_name}) in game {game_id}")
    
        print(f"✅ done processing {total_counter} files!")
        total_counter += 1

        # ✅ Commit playing elevens per game to avoid duplicate insert errors
        if playing_elevens_to_add:
            session.bulk_save_objects(playing_elevens_to_add)
            session.flush()
            session.commit()
            print(f"✅ Inserted {len(playing_elevens_to_add)} playing eleven records")
            # Update in-memory tracker
            for pe in playing_elevens_to_add:
                key = (pe.game_id, pe.player_id, pe.team_id)
                existing_playing_elevens.add(key)
            playing_elevens_to_add.clear()

        # Commit batched inserts:
        if players_to_add:
            session.bulk_save_objects(players_to_add)
            session.flush()
            players_to_add.clear()
            print(f"✅ Inserted {len(players_to_add)} players")
        
        if matches_to_add:
            session.bulk_save_objects(matches_to_add)
            print(f"✅ Inserted {len(matches_to_add)} matches")
            session.commit()
            print("Committed session...")

    # print completion statement
    print("✅ Done adding data to players, matches, and playing_elevens tables!")


def add_ballbyball(session, game_ids):
    # pre load player and team maps:
    players_map = {p.cricsheet_id: p.id for p in session.query(Player).all()}
    teams_map = {t.team_name: t.id for t in session.query(Team).all()}

    # pre load existing game_ids within the ball_by_ball table:
    existing_ballbyball_games = {
        g[0] for g in session.query(BallByBall.game_id).distinct().all()
    }
    
    # keep count of total inserted, and total games
    total_inserted = 0
    total_games = 0
    
    # loop thru each game_id:
    for game_id in game_ids:
        # skip game if already processed:
        if game_id in existing_ballbyball_games:
            print(f"⏭️ Game {game_id} already has ball-by-ball data, skipping...")
            continue
        
        # get the filename and the relative filepath
        filename = f"{game_id}.json"
        filepath = os.path.join(folder_path, filename)
        
        # Raise error if file is not found and exit:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ File not found: {filename}")
    
        # Load the json file's data:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # extract appropriate data:
        innings = data.get("innings", [])
        registry = data.get("info", {}).get("registry", {}).get("people", {})
        ball_by_ball_rows = []
        inning_number = 1
        
        # for each inning in the game...
        for inning in innings:
            batting_team_name = inning.get("team")
            batting_team_id = teams_map.get(batting_team_name)
            ball_number = 1
            total_runs = 0
            total_wickets = 0
            
            # for each over in a given innings:
            for over in inning.get("overs", []):
                # for each delivery in that over:
                for delivery in over.get("deliveries", []):
                    striker_name = delivery.get("batter")
                    non_striker_name = delivery.get("non_striker")
                    bowler_name = delivery.get("bowler")

                    striker_cricsheet_id = registry.get(striker_name)
                    non_striker_cricsheet_id = registry.get(non_striker_name)
                    bowler_cricsheet_id = registry.get(bowler_name)
                    
                    striker_id = players_map.get(striker_cricsheet_id)
                    non_striker_id = players_map.get(non_striker_cricsheet_id)
                    bowler_id = players_map.get(bowler_cricsheet_id)
                    
                    runs = delivery.get("runs", {})
                    batter_runs = runs.get("batter", 0)
                    extras_runs = runs.get("extras", 0)
                    runs_this_ball = batter_runs + extras_runs
                    total_runs += runs_this_ball
                    
                    # check for wicket:
                    wicket_info = delivery.get("wickets", [])
                    if wicket_info:
                        wicket = 1
                        wicket_type = wicket_info[0].get("kind")
                        total_wickets += 1
                    else:
                        wicket = 0
                        wicket_type = None
                    
                    # check for extras type:
                    extras_dict = delivery.get("extras", {})
                    if extras_dict:
                        extras_type = list(extras_dict.keys())[0]
                    else:
                        extras_type = None
                    
                    # create ballbyball object:
                    row = BallByBall(
                        game_id=game_id,
                        inning=inning_number,
                        ball_number=ball_number,
                        batting_team=batting_team_id,
                        striker_id=striker_id,
                        nonstriker_id=non_striker_id,
                        bowler_id=bowler_id,
                        batter_runs=batter_runs,
                        extras_runs=extras_runs,
                        extras_type=extras_type,
                        wicket=wicket,
                        wicket_type=wicket_type,
                        total_team_runs=total_runs,
                        total_team_wickets=total_wickets
                    )

                    # append row to list:
                    ball_by_ball_rows.append(row)
                    # increment ball number:
                    ball_number += 1
            # increment inning number:
            inning_number += 1
        
        # after each game, commit all rows that have been recorded:
        if ball_by_ball_rows:
            session.bulk_save_objects(ball_by_ball_rows)
            session.flush()
            session.commit()
            print(f"✅ Inserted {len(ball_by_ball_rows)} ball-by-ball entries for game {game_id}")
            total_inserted += len(ball_by_ball_rows)
        else:
            print("NO ROWS DETECTED FOR THIS GAME!!!")
        
        total_games += 1
        print(f"✅ Done processing {total_games} games!")
        
    # print completion statement:
    print(f"✅ Done adding ball-by-ball data. Total rows inserted: {total_inserted}")














# MAIN:
def main():
    print("Code execution starting...")
    try:
        initialize_db_params()
        
        create_all_tables()
        load_all_game_ids()
        
        # Try to create a session and connect to the DB
        session = Session()
        session.execute(text('SELECT 1'))  # optional: quick sanity check

        print("✅ Database session started successfully.")

        # get list of all game_ids:
        list_of_game_ids = load_all_game_ids()
        
        # add data to master tables (teams, venues, officials, match_formats):
        add_matchformats_venues_teams_officials(session, list_of_game_ids)
        
        # add data to more tables (players, matches, playing_elevens):
        add_players_matches_playingelevens(session, list_of_game_ids)
        
        # add data to the ball by ball table:
        add_ballbyball(session, list_of_game_ids)

        session.commit()
        
    except OperationalError as e:
        print("❌ OperationalError: Could not connect to the database.")
        print(e)

    except SQLAlchemyError as e:
        print("❌ SQLAlchemyError: Issue with the database session or query.")
        print(e)

    except Exception as e:
        print("❌ Unexpected error during main execution.")
        print(e)

    finally:
        try:
            session.close()
            print("🔒 Session closed.")
        except NameError:
            print("No session was created, so nothing to close.")
        except Exception as e:
            print("⚠️ Error while closing the session:", e)
            

if __name__ == "__main__":
    main()

