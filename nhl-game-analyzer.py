import requests
import pandas as pd
from pandas import json_normalize
from datetime import datetime, timedelta
import os
import datarobotx as drx
import pytz
import streamlit as st
import snowflake.connector
import re


# Configure the streamlit page title, favicon, layout, etc
st.set_page_config(page_title="NHL Picks", layout="wide")


#@st.cache(suppress_st_warning=True, show_spinner=False)
def getPredictionsFromSnowflake():
    # Connect to Snowflake. Replace the values below to match your environment
    ctx = snowflake.connector.connect(
        user="DATAROBOT",
        password='D@t@robot',
        account="datarobot_partner",
        warehouse="DEMO_WH",
        database="SANDBOX",
        schema="NHL_GAME_DATA",
    )
    # Create a cursor object.
    cur = ctx.cursor()

    # Execute a statement that will generate a result set.
    sql = '''
    SELECT *
    FROM "PREDICTIONS";
    '''
    cur.execute(sql)
    # Fetch the result set from the cursor and deliver it as the Pandas DataFrame.
    predictions = cur.fetch_pandas_all()
    predictions.drop_duplicates(inplace=True)
    return predictions

#@st.cache(suppress_st_warning=True, show_spinner=False)
def getPastPredictionsFromSnowflake():
    # Connect to Snowflake. Replace the values below to match your environment
    ctx = snowflake.connector.connect(
        user=os.environ["snowflakeUser"],
        password=os.environ["snowflakePassword"],
        account=os.environ["snowflakeAccount"],
        warehouse=os.environ["snowflakeWarehouse"],
        database=os.environ["snowflakeDatabase"],
        schema=os.environ["snowflakeSchema"],
    )
    # Create a cursor object.
    cur = ctx.cursor()

    sql = '''
            SELECT *
            FROM "PAST_PREDICTIONS_VS_ACTUALS"
            ORDER BY TO_TIMESTAMP("startTimeUTC") DESC
            '''
    cur.execute(sql)
    pastPredictions = cur.fetch_pandas_all()
    pastPredictions["Winner"] = "None"
    pastPredictions.loc[pastPredictions["homeTeam_score"] > pastPredictions["awayTeam_score"], "Winner"] = "Home"
    pastPredictions.loc[pastPredictions["homeTeam_score"] < pastPredictions["awayTeam_score"], "Winner"] = "Away"
    pastPredictions["Correct Prediction"] = "None"
    pastPredictions.loc[
        pastPredictions["Winner"] == pastPredictions["Winner_PREDICTION"], "Correct Prediction"] = "Correct"
    pastPredictions.loc[
        pastPredictions["Winner"] != pastPredictions["Winner_PREDICTION"], "Correct Prediction"] = "Incorrect"
    pastPredictions.loc[pastPredictions["Winner"] == "None", "Correct Prediction"] = "N/A"
    pastPredictions["startTimeUTC"] = pd.to_datetime(pastPredictions["startTimeUTC"]).dt.date
    pastPredictions = pastPredictions[
        ["ID", "startTimeUTC", "awayTeam_name_default", "awayTeam_score", "homeTeam_name_default", "homeTeam_score",
         "Winner", "Winner_PREDICTION", "Correct Prediction"]]
    pastPredictions.columns = ['Game id', 'Game Day', 'Away Team', 'Away Score', 'Home Team', 'Home Score',
                               'Actual Winner', 'Pre-Game Prediction', 'Correct Prediction']
    pastPredictions.drop_duplicates(inplace=True)
    return pastPredictions


# Function to extract corresponding homeTeam and awayTeam feature names
def get_feature_names(feature):
    if feature.startswith("Diff_"):
        home_feature = "homeTeam" + feature[4:]
        away_feature = "awayTeam" + feature[4:]
    elif feature.startswith("homeTeam_"):
        home_feature = feature
        away_feature = "awayTeam" + feature[8:]
    elif feature.startswith("awayTeam_"):
        home_feature = "homeTeam" + feature[8:]
        away_feature = feature
    else:
        home_feature = away_feature = None
    return home_feature, away_feature

def prepPredictionExplanations(game):
    game.drop_duplicates("ID", inplace=True)
    gameSorted = game.copy()
    gameSorted.columns = [col.lower() for col in gameSorted.columns]
    if game["Winner_PREDICTION"].iloc[0] == "Home":
        gameSorted = drx.melt_explanations(gameSorted).sort_values("strength", ascending=False)
    else:
        gameSorted = drx.melt_explanations(gameSorted).sort_values("strength", ascending=True)

    gameSorted['Away Team'] = None
    gameSorted['Home Team'] = None

    for index, row in gameSorted.iterrows():
        try:
            home_feature, away_feature = get_feature_names(row['feature_name'])

            # If valid feature names are found, assign values from game_df
            if home_feature and away_feature:
                gameSorted.at[index, 'Away Team'] = game[away_feature].iloc[0]
                gameSorted.at[index, 'Home Team'] = game[home_feature].iloc[0]
        except:
            pass
    gameSorted = gameSorted[["strength", "feature_name", "actual_value", "qualitative_strength", "Away Team", "Home Team"]]
    gameSorted.columns = ["Strength", "Statistic","Value","Impact to Home Team", "Away Team", "Home Team"]
    return gameSorted

def explainPredictionDR(game, matchup):

    gameSorted = prepPredictionExplanations(game)

    prompt = "Home team: " + str(game["homeTeam_standings_teamName_default"].iloc[0]) +" AI Model's Home team win probability: " + str(game["Winner_Home_PREDICTION"].iloc[0]) + "     Away team: " + "     Data: " + str(
        game["awayTeam_standings_teamName_default"].iloc[0]) +" AI Model's Away team win probability: " + str(game["Winner_Away_PREDICTION"].iloc[0]) + gameSorted.to_json(orient='records')
    data = pd.DataFrame({"promptText": [prompt]})
    API_URL = 'https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = os.environ["DATAROBOT_API_TOKEN"]
    DATAROBOT_KEY = os.environ["DATAROBOT_KEY"]
    deployment_id = '6571ff84f2970c0f7bd88169'
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    predictions_response = requests.post(
        url,
        data=data.to_json(orient='records'),
        headers=headers
    )
    return predictions_response.json()["data"][0]["prediction"]

def beautify_metric_name_final(name):
    # Replace specific abbreviations and terms
    name = (name.replace('Abbrev', 'Abbreviation')
            .replace('standings', '')
            .replace('abbrev', 'Abbreviation')
            .replace('Ot', 'Over Time')
            .replace('Pctg', 'Percentage')
            .replace('sog', 'Shots on Goal')
            .replace('pim', 'Penalty Minutes')
            .replace('rolling10', 'Rolling 10 Games'))

    # Replace 'l10' with 'Last 10'
    name = name.replace('l10', 'Last 10').replace('L10', 'Last 10')

    # Replace underscores with spaces
    name = name.replace('_', ' ')

    # Splitting based on uppercase letters and numbers, adding spaces
    words = []
    current_word = ''
    for char in name:
        if char.isupper() or (char.isdigit() and current_word and not current_word[-1].isdigit()):
            if current_word:
                words.append(current_word)
            current_word = char
        else:
            current_word += char
    words.append(current_word)  # Add the last word

    # Join words with spaces and convert to title case
    cleaned_name = ' '.join(words).title()

    # Use regular expressions to remove unnecessary whitespace
    # This will replace multiple spaces with a single space and strip leading/trailing spaces
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()

    return cleaned_name


def mainPage():
    eastern = pytz.timezone('US/Eastern')
    startdate = datetime.now(eastern).date() - timedelta(days=1)
    enddate = datetime.now(eastern).date() + timedelta(days=2)
    with st.spinner("processing..."):
        predictions = getPredictionsFromSnowflake()
        predictions["Game Name"] = predictions["awayTeam_standings_teamName_default"].astype(str) + " @ " + predictions["homeTeam_standings_teamName_default"].astype(str)
        print(predictions)
    with st.sidebar:
        #date = datetime.now(eastern).date()
        date = predictions["startTimeUTC"].max()
        print(f"the date is {date}")
        gameDayPredictions = predictions.loc[predictions["startTimeUTC"].astype(str) == str(date)]
        gameChoice = st.selectbox(label="Tonight's Games", options=gameDayPredictions["Game Name"].unique())
        # gameChoice = gameDayPredictions["Game Name"].unique()[0]
        game = predictions.loc[predictions["Game Name"] == gameChoice]
        game.drop_duplicates(["ID"], inplace=True)
        print("Game:")
        print(game)


    # Title
    titleContainer = st.container()
    titleContainer1, titleContainer2, titleContainer3 = titleContainer.columns([0.1, 0.3, 1])
    # titleContainer1.image("NHL-Logo-700x394.png", width = 75)

    # Predicted Winner
    if game["Winner_PREDICTION"].iloc[0] == "Home":
        predictedWinner = game["homeTeam_standings_teamCommonName_default"].iloc[0]
        predictionBadgeHome = " :money_with_wings: "
        predictionBadgeAway = ""
        predictionHome = f"Odds favor the {predictedWinner} :money_with_wings:"
        predictionAway = " "
    else:
        predictedWinner = game["awayTeam_standings_teamCommonName_default"].iloc[0]
        predictionBadgeHome = ""
        predictionBadgeAway = " :money_with_wings: "
        predictionAway = f"Odds favor the {predictedWinner} :money_with_wings:"
        predictionHome = " "

    head2head, allGames = st.tabs(["Head-to-Head", "All Game Predictions"])
    with head2head:
        # 2 columns with selected team logos
        container1 = st.container()
        awayCol, middleCol, homeCol = container1.columns([1, 0.85, 1])
        awayCol.title(game["awayTeam_standings_teamName_default"].iloc[0])
        awayCol.image(str(game["awayTeam_logo"].iloc[0]), width=275)
        middleCol.title("           ")
        middleCol.title("           ")
        middleCol.title("           ")
        middleCol.image("vs-image.png", width=150)
        homeCol.title(game["homeTeam_standings_teamName_default"].iloc[0])
        homeCol.image(str(game["homeTeam_logo"].iloc[0]), width=275)
        container2 = st.container()
        container2Left, container2Mid, container2Right = container2.columns([.5, 1, .5])

        # Display the prediction with money wings

        container2.header(" ")
        if abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.02:
            container2.header(
                f"By the numbers, it's a very close match up. The {predictedWinner} do have a slight edge :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.03:
            container2.header(f"The data suggests it's close but the {predictedWinner} have an edge :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.04:
            container2.header(f"It's a good matchup but the {predictedWinner} have an edge :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.05:
            container2.header(f"The {predictedWinner} have an edge according to the data :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.06:
            container2.header(f"The {predictedWinner} have a clear edge according to the data :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.07:
            container2.header(f"The {predictedWinner} have an advantage by the numbers :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.08:
            container2.header(f"The {predictedWinner} have a clear advantage in the data :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.09:
            container2.header(f"The {predictedWinner} have a solid advantage :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.09:
            container2.header(f"The {predictedWinner} will probably take this one :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.1:
            container2.header(f"The {predictedWinner} will most likely take this one :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.12:
            container2.header(f"The {predictedWinner} will probably get the W :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) <= 0.15:
            container2.header(
                f"Looks like the {predictedWinner} have a big advantage here and should take the win :money_with_wings:")
        elif abs(game["Winner_Home_PREDICTION"].iloc[0] - game["Winner_Away_PREDICTION"].iloc[0]) >= 0.15:
            container2.header(f"The {predictedWinner} have a significant advantage :money_with_wings:")

        # 2 tables with head-to-head key metrics from standings
        # Filtering and pivoting for homeTeam
        game.columns = [col.replace('homeTeam.', 'homeTeam_') for col in game.columns]
        game.columns = [col.replace('awayTeam.', 'awayTeam_') for col in game.columns]

        homeTeam_cols = [col for col in game.columns if col.startswith("homeTeam_")]
        homeTeam_df = game[homeTeam_cols]
        homeTeam_df["Win Probability"] = game["Winner_Home_PREDICTION"]
        homeTeam_df = homeTeam_df.T
        homeTeam_df.index = homeTeam_df.index.str.replace("homeTeam_", "")

        # Filtering and pivoting for awayTeam
        awayTeam_cols = [col for col in game.columns if col.startswith("awayTeam_")]
        awayTeam_df = game[awayTeam_cols]
        awayTeam_df["Win Probability"] = game["Winner_Away_PREDICTION"]
        awayTeam_df = awayTeam_df.T
        awayTeam_df.index = awayTeam_df.index.str.replace("awayTeam_", "")

        # Combine into single table
        matchup = pd.concat([awayTeam_df, homeTeam_df], axis=1)
        matchup.columns = ["Away", "Home"]
        matchup.columns = [matchup["Away"].loc["standings_teamName_default"], matchup["Home"].loc["standings_teamName_default"]]
        matchup.drop(['id', 'logo', 'darkLogo', 'radioLink',
                      'placeName_default', 'standings_waiversSequence',
                      'standings_wildcardSequence', 'standings_teamName_fr'], axis=0, inplace=True)

        matchup.index = matchup.index.map(beautify_metric_name_final)

        getAnalysisButton = st.button(label="Explain it !")
        if getAnalysisButton:
            with st.spinner("Thinking..."):
                explanation = explainPredictionDR(game,matchup)

            tab1, tab2 = st.tabs(["Why", "Model Reason Codes"])
            with tab1:
                # GPT Don Cherry explanation of who the winner will likely be
                try:
                    st.write(explanation)
                except Exception as e:
                    st.write("Explanation is unavailable at the moment.")
            with tab2:
                st.dataframe(prepPredictionExplanations(game))

        with st.expander("See details"):
            container3 = st.container()
            container3.header("Head-to-Head")
            container3.subheader("League Matchup")
            container3.table(matchup.loc[["Win Probability", "Wins", "Win Percentage", "Losses", "League Sequence",
                                          "League Home Sequence", "League Road Sequence", "League Last 10 Sequence",
                                          "Corsi Rolling 10 Games", "Shots On Goal Rolling 10 Games",
                                          "Blocks Rolling 10 Games", "Hits Rolling 10 Games",
                                          "Power Play Conversion Rolling 10 Games", "Road Wins", "Road Losses",
                                          "Home Wins", "Home Losses"]])
            container3.subheader("Conference Matchup")
            container3.table(matchup.loc[["Conference Name", "Conference Sequence", "Conference Home Sequence",
                                          "Conference Road Sequence", "Conference Last 10 Sequence"]])
            container3.subheader("Division Matchup")
            container3.table(matchup.loc[["Division Name", "Division Sequence", "Division Home Sequence",
                                          "Division Road Sequence", "Division Last 10 Sequence"]])
            container3.subheader("Other Key Metrics")
            container3.table(matchup.drop(
                ["Abbreviation", "Odds", "Conference Abbreviation", "Conference Home Sequence",
                 "Conference Last  10 Sequence",
                 "Conference Name", "Conference Road Sequence", "Conference Sequence", "Division Abbreviation",
                 "Division Home Sequence", "Division Last  10 Sequence", "Division Name",
                 "Division Road Sequence", "Division Sequence",
                 "Score", "Place Name Fr", "Game Type Id", "Home Ties", "Season Id"],
                axis="rows", errors='ignore'))

    with allGames:
        st.header("All Game Predictions")
        st.caption(
            "Completed games show the pre-game probabilities and prediction against the final outcome. Sort the table by probability to get the best bets.")
        predictions = predictions.loc[:, ~predictions.columns.duplicated()]
        # st.dataframe(predictions)
        st.subheader("Past Predictions")
        pastPredictions = getPastPredictionsFromSnowflake()
        st.dataframe(pastPredictions)
        st.subheader("Tonight's Game Predictions")
        currentPredictions = getPredictionsFromSnowflake()
        currentPredictions.drop_duplicates("ID", inplace=True)
        currentPredictions = currentPredictions[
            ["ID", "startTimeUTC", "awayTeam_standings_teamName_default", "homeTeam_standings_teamName_default", "Winner_Away_PREDICTION",
             "Winner_Home_PREDICTION", "Winner_PREDICTION"]]
        currentPredictions.columns = ['Game id', 'Game Day', 'Away Team', 'Home Team', 'Probability Away Wins',
                                      'Probability Home Wins', 'AI Prediction']
        st.dataframe(currentPredictions)



# Make Predictions
# eastern = pytz.timezone('US/Eastern')
# startdate=datetime.now(eastern).date() - timedelta(days=30)
# enddate=datetime.now(eastern).date() + timedelta(days=1)
# predictions = getPredictions(startdate=startdate, enddate=enddate)


# Main app
def _main():
    hide_streamlit_style = """
    <style>
    # MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)  # This let's you hide the Streamlit branding
    mainPage()


if __name__ == "__main__":
    _main()
