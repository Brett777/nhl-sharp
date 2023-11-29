import requests
import pandas as pd
from pandas import json_normalize
from datetime import datetime, timedelta
import os
import datarobotx as drx
import pytz
import streamlit as st
from openai import OpenAI
import plotly.graph_objects as go

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
drx.Context(endpoint=os.environ["DATAROBOT_ENDPOINT"], token=os.environ["DATAROBOT_API_KEY"])

#Configure the streamlit page title, favicon, layout, etc
st.set_page_config(page_title="NHL Picks", layout="wide")

# Data prep function to convert string ratio to decimal value.
@st.cache_data(show_spinner=False)
def convert_to_decimal(ratio_str):
    try:
        numerator, denominator = ratio_str.split('/')
        if denominator == '0':
            return 0  # Avoid division by zero
        return float(numerator) / float(denominator)
    except:
        pass
    return 0
@st.cache_data(show_spinner=False)
def get_nhl_schedule(start_date, end_date, train):
    # Convert dates to datetime objects
    # start_date = datetime.strptime(start_date, '%Y-%m-%d')
    # end_date = datetime.strptime(end_date, '%Y-%m-%d')

    all_games = []

    current_date = start_date
    while current_date <= end_date:
        # Format the current date in the required format
        date_str = current_date.strftime('%Y-%m-%d')
        print(date_str)
        # Make the request to the NHL API
        response = requests.get(f"https://api-web.nhle.com/v1/schedule/{date_str}")

        if response.status_code == 200:
            # If the request is successful, extract the games data
            schedule = response.json()
            for game_week in schedule['gameWeek']:
                for game in game_week['games']:
                    all_games.append(game)

        # Move to the next week
        current_date += timedelta(days=7)

    # Create a DataFrame from the collected games
    df = pd.DataFrame(all_games)

    # Convert startTimeUTC to Eastern Time
    eastern = pytz.timezone('US/Eastern')
    df['startTimeUTC'] = pd.to_datetime(df['startTimeUTC']).dt.tz_convert(eastern)
    df['startTimeUTC'] = pd.to_datetime(df['startTimeUTC']).dt.date

    # Flatten homeTeam and awayTeam
    df = df.join(pd.json_normalize(df['homeTeam']).add_prefix('homeTeam_'))
    df = df.join(pd.json_normalize(df['awayTeam']).add_prefix('awayTeam_'))

    # Drop unnecessary columns
    columns_to_drop = ['tvBroadcasts', 'periodDescriptor', 'gameOutcome',
                       'winningGoalie', 'winningGoalScorer', 'threeMinRecap',
                       'gameCenterLink', 'threeMinRecapFr', 'ticketsLink',
                       'homeTeam', 'awayTeam']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    if train == True:
        df["Winner"] = "Away"
        df.loc[df["homeTeam_score"]>df["awayTeam_score"], "Winner"] = "Home"
    return df

def getBoxscore(gameID):
    #gameID = "2023020267"
    # URL of the API endpoint
    url = "https://api-web.nhle.com/v1/gamecenter/"+str(gameID)+"/boxscore"

    # Sending a GET request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response into a DataFrame
        json_data = response.json()
        # Flatten the nested dictionaries
        flat_game = json_normalize(json_data)

        # Handling lists within the game item, if any
        # We'll convert lists to a string representation for simplicity
        for column in flat_game.columns:
            if isinstance(flat_game.iloc[0][column], list):
                flat_game[column] = flat_game[column].apply(lambda x: ', '.join(map(str, x)) if x else '')

        return flat_game

    return response.status_code

def getBoxscoreTable(schedule):
    boxscores = pd.DataFrame()
    for id in schedule["id"]:
        boxscore = getBoxscore(id)
        print(id)
        try:
            boxscores = pd.concat([boxscores, boxscore], axis=0)
        except:
            pass
    boxscores = boxscores[['id','gameDate', 'awayTeam.name.default', 'awayTeam.abbrev', 'awayTeam.score',
                                         'awayTeam.sog', 'awayTeam.faceoffWinningPctg',
                                         'awayTeam.powerPlayConversion', 'awayTeam.pim', 'awayTeam.hits',
                                         'awayTeam.blocks', 'awayTeam.logo', 'homeTeam.id',
                                         'homeTeam.name.default', 'homeTeam.abbrev', 'homeTeam.score',
                                         'homeTeam.sog', 'homeTeam.faceoffWinningPctg',
                                         'homeTeam.powerPlayConversion', 'homeTeam.pim', 'homeTeam.hits',
                                         'homeTeam.blocks', 'homeTeam.logo']].copy()
    boxscores['awayTeam.powerPlayConversion'] = boxscores['awayTeam.powerPlayConversion'].apply(convert_to_decimal)
    boxscores['homeTeam.powerPlayConversion'] = boxscores['homeTeam.powerPlayConversion'].apply(convert_to_decimal)
    boxscores['homeWin'] = 0
    boxscores.loc[boxscores['awayTeam.score'] > boxscores['homeTeam.score'], "homeWin"] = "1"
    boxscores.drop(['awayTeam.score', 'homeTeam.score'], axis='columns', inplace=True)
    return boxscores

def joinBoxscore(schedule):
    boxscores = pd.DataFrame()
    for id in schedule["id"]:
        boxscore = getBoxscore(id)
        print(id)
        boxscores = pd.concat([boxscores,boxscore], axis=0)
    schedule = schedule.merge(boxscores[['id', 'awayTeam.name.default', 'awayTeam.abbrev', 'awayTeam.score',
       'awayTeam.sog', 'awayTeam.faceoffWinningPctg',
       'awayTeam.powerPlayConversion', 'awayTeam.pim', 'awayTeam.hits',
       'awayTeam.blocks', 'awayTeam.logo', 'homeTeam.id',
       'homeTeam.name.default', 'homeTeam.abbrev', 'homeTeam.score',
       'homeTeam.sog', 'homeTeam.faceoffWinningPctg',
       'homeTeam.powerPlayConversion', 'homeTeam.pim', 'homeTeam.hits',
       'homeTeam.blocks', 'homeTeam.logo']], how="left", on="id")

    return schedule
@st.cache_data(show_spinner=False)
def getStandings(schedule):
    standings = pd.DataFrame()
    for date in schedule["startTimeUTC"]:
        # URL of the API endpoint
        print(date)
        url = "https://api-web.nhle.com/v1/standings/" + str(date)
        # Sending a GET request to the API
        response = requests.get(url)

        # Parse the JSON response into a DataFrame
        json_data = response.json()
        standing = pd.json_normalize(json_data['standings'])
        standings = pd.concat([standings, standing], axis=0)
    return standings
@st.cache_data(show_spinner=False)
def fill_missing_dates(df, date_col, team_col):
    """
    Fills missing dates in a dataframe for each team with the most recent available data.
    Handles duplicate date entries by keeping the last occurrence.

    Parameters:
    df (DataFrame): The dataframe to process.
    date_col (str): The name of the column containing dates.
    team_col (str): The name of the column uniquely identifying each team.

    Returns:
    DataFrame: A dataframe with missing dates filled in.
    """
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Identify the date range
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

    # Identify all unique teams
    teams = df[team_col].unique()

    # Initialize a list to hold dataframes for each team
    team_dfs = []

    for team in teams:
        # Filter the dataframe for the current team
        team_df = df[df[team_col] == team]

        # Drop duplicate dates, keeping the last occurrence
        team_df = team_df.drop_duplicates(subset=date_col, keep='last')

        # Reindex the dataframe to have all dates in the range, filling missing ones
        team_df = team_df.set_index(date_col).reindex(all_dates)

        # Forward fill the missing values
        team_df.fillna(method='ffill', inplace=True)

        # Reset the index to turn the dates back into a column
        team_df.reset_index(inplace=True)

        # Rename the 'index' column back to the original date column name
        team_df.rename(columns={'index': date_col}, inplace=True)

        # Add the processed dataframe to the list
        team_dfs.append(team_df)

    # Concatenate all team dataframes
    filled_df = pd.concat(team_dfs, ignore_index=True)

    return filled_df
@st.cache_data(show_spinner=False)
def joinStandings(schedule, standings, train):
    homeTeamStandings = standings.copy()
    homeTeamStandings.columns = ['homeTeam_' + col for col in homeTeamStandings.columns]
    homeTeamStandings = fill_missing_dates(homeTeamStandings, "homeTeam_date", "homeTeam_teamAbbrev.default")
    awayTeamStandings = standings.copy()
    awayTeamStandings.columns = ['awayTeam_' + col for col in awayTeamStandings.columns]
    awayTeamStandings = fill_missing_dates(awayTeamStandings, "awayTeam_date", "awayTeam_teamAbbrev.default")
    schedule["startTimeUTC"] = pd.to_datetime(schedule["startTimeUTC"])
    if train == True:
        homeTeamStandings['homeTeam_date'] += pd.Timedelta(days=1)
        awayTeamStandings['awayTeam_date'] += pd.Timedelta(days=1)
    schedule = pd.merge(schedule, homeTeamStandings, how="left", left_on=["startTimeUTC","homeTeam_abbrev"], right_on=["homeTeam_date","homeTeam_teamAbbrev.default"])
    schedule = pd.merge(schedule, awayTeamStandings, how="left", left_on=["startTimeUTC", "awayTeam_abbrev"], right_on=["awayTeam_date", "awayTeam_teamAbbrev.default"])
    #schedule.drop(['homeTeam_placeName.default', 'awayTeam_placeName.default'], axis='columns', inplace=True)
    return schedule


# Training Data
# df = get_nhl_schedule("2017-10-01", "2023-11-14", train=True)
# df3 = getStandings(df)
# df = joinStandings(df,df3, train=True)
# print("done")
# df.to_csv("schedule.csv", index=False)
# print("done")

@st.cache_data(show_spinner=False)
def getPredictions(startdate, enddate):
    df = get_nhl_schedule(startdate, enddate, train=False)
    df = df.loc[df["startTimeUTC"]<enddate]
    df2 = getStandings(df)
    df3 = joinStandings(df, df2, train=False)
    deployment = drx.Deployment("6560f92ef3aa6cdc30c695a1")
    predictions = drx.Deployment.predict(deployment, X=df3, max_explanations=6)
    predictions = pd.DataFrame(predictions)
    probabilities = drx.Deployment.predict_proba(deployment, X=df3)
    probabilities = pd.DataFrame(probabilities)
    probabilities.columns = ["Home Team Win Probability", "Away Team Win Probability"]
    predictions = pd.concat([predictions,probabilities,df3], axis=1)
    return predictions

def explainPrediction(game):
    completion = client.chat.completions.create(
        model='gpt-4-1106-preview',
        #model="gpt-4",
        #model="gpt-3.5-turbo",
        temperature=1,
        messages=[
            {"role": "system",
             "content": """
                            You are Don Cherry.
                            Speak in the voice of Don Cherry and do not break character.
                            The user will ask for your assistance in explaining which team will likely win the hockey game. 
                            The user will provide data for you to analyze.
                            
                            Here's how to interpret the data:
                            explanation_x_feature_name is a data attribute from the standings table. Sometimes your audience won't understand what these mean, so you'll have to explain it, ELI5 style.Do not reference the feature name with quotes and underscores. Instead use the proper term for it. 
                            explanation_x_strength is the amount this feature impacts the prediction. Never explicitly say this value, but factor it into your analysis.
                            explanation_x_value is the actual value of the feature.
                            explanation_x_qualitative_strength is the amount this feature impacts the prediction. Never explicitly say this value, but factor it into your analysis.                            
                                 
                            """
             },
            {"role": "user", "content": "Home team: " + str(game["homeTeam_teamName.default"].iloc[0])
             + " Away team: " + str(game["awayTeam_teamName.default"].iloc[0])
             + game.iloc[:,:27].to_json(orient='records')}
        ]
    )
    return completion.choices[0].message.content

def createGuage():
    # Create figure
    fig = go.Figure()

    # Configure the gauge chart
    fig.add_trace(go.Indicator(
        mode="gauge+number",  # Gauge chart with a numeric display
        value=70,  # Current value (e.g., progress percentage)
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Project Progress"},  # Title of the gauge chart
        gauge={
            'axis': {'range': [None, 100]},  # Setting the range of the gauge from 0 to 100
            'bar': {'color': "blue"},  # Color of the gauge's bar
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},  # Color settings for different ranges
                {'range': [50, 100], 'color': "gray"}  # Customizing colors based on ranges
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},  # Setting the threshold line properties
                'thickness': 0.75,
                'value': 90  # Threshold value
            }
        }
    ))

    return fig

def mainPage():
    eastern = pytz.timezone('US/Eastern')
    startdate=datetime.now(eastern).date() - timedelta(days=1)
    enddate=datetime.now(eastern).date() + timedelta(days=1)
    #gameChoice = "2023-11-27 Florida Panthers @ Ottawa Senators"
    with st.spinner("processing..."):
        predictions = getPredictions(startdate=startdate, enddate=enddate)
        predictions["Game Name"] = predictions["startTimeUTC"].astype(str) + " " + predictions["awayTeam_teamName.default"].astype(str) + " @ " + predictions["homeTeam_teamName.default"].astype(str)
        print(predictions)
    with st.sidebar:
        gameChoice = st.selectbox(label="Game", options=predictions["Game Name"].unique())
        game = predictions.loc[predictions["Game Name"]==gameChoice]
        print(game)

    # Predicted Winner
    if game["prediction"].iloc[0] == "Home":
        predictedWinner = game["homeTeam_teamName.default"].iloc[0]
        predictionBadgeHome = " :money_with_wings: "
        predictionBadgeAway = ""
    else:
        predictedWinner = game["awayTeam_teamName.default"].iloc[0]
        predictionBadgeHome = ""
        predictionBadgeAway = " :money_with_wings: "

    # create the gauge
    fig = createGuage()
    # 2 columns with selected team logos
    container1 = st.container()
    awayCol, middleCol, homeCol = container1.columns([1,0.5,1])
    awayCol.title(predictionBadgeAway + str(game["awayTeam_teamName.default"].iloc[0]) + predictionBadgeAway)
    awayCol.image(str(game["awayTeam_logo"].iloc[0]), width = 275)
    middleCol.title("           ")
    middleCol.title("           ")
    #st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    middleCol.image("vs-image.png", width=150)
    homeCol.title(predictionBadgeHome + str(game["homeTeam_teamName.default"].iloc[0]) + predictionBadgeHome)
    homeCol.image(str(game["homeTeam_logo"].iloc[0]), width = 275)
    container2 = st.container()
    container2Left,container2Mid,container2Right = container2.columns([.5, 1, .5])

    if game["gameState"].iloc[0] == "FUT":
        st.header("Coach's Analysis")
        st.caption(str(game["startTimeUTC"].iloc[0].date()))
    else:
        st.header("Score")
        st.subheader(str(game["homeTeam_teamName.default"].iloc[0]) + ": " + str(game["homeTeam_score"].iloc[0].astype(int)))
        st.subheader(str(game["awayTeam_teamName.default"].iloc[0]) + ": " + str(game["awayTeam_score"].iloc[0].astype(int)))
    getAnalysisButton = st.button(label="Explain it Coach!")
    if getAnalysisButton:
        with st.spinner("Don Cherry is thinking..."):
            explanation = explainPrediction(game)

        # GPT Don Cherry explanation of who the winner will likely be
        try:
            st.write(explanation)
        except Exception as e:
            st.write("Don Cherry explanation is unavailable at the moment.")

    # 2 tables with head-to-head key metrics from standings
    # Filtering and pivoting for homeTeam
    homeTeam_cols = [col for col in game.columns if col.startswith("homeTeam_")]
    homeTeam_df = game[homeTeam_cols].T
    homeTeam_df.index = homeTeam_df.index.str.replace("homeTeam_","")

    # Filtering and pivoting for awayTeam
    awayTeam_cols = [col for col in game.columns if col.startswith("awayTeam_")]
    awayTeam_df = game[awayTeam_cols].T
    awayTeam_df.index = awayTeam_df.index.str.replace("awayTeam_", "")

    # Combine into single table
    matchup = pd.concat([awayTeam_df,homeTeam_df], axis=1)
    matchup.columns = ["Away","Home"]
    matchup.columns = [matchup["Away"].loc["teamName.default"], matchup["Home"].loc["teamName.default"]]
    container3 = st.container()
    container3.table(matchup.iloc[:, 10:-6])





#Main app
def _main():
    hide_streamlit_style = """
    <style>
    # MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) # This let's you hide the Streamlit branding
    mainPage()

if __name__ == "__main__":
    _main()
