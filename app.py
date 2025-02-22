
import streamlit as st
import pandas as pd
import pickle
import json
import hashlib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to load credentials
def load_credentials():
    with open('credentials.json', 'r') as file:
        return json.load(file)

# Function to save credentials
def save_credentials(credentials):
    with open('credentials.json', 'w') as file:
        json.dump(credentials, file)

# Function to authenticate user
def authenticate(username, password):
    credentials = load_credentials()
    for user in credentials['users']:
        if user['username'] == username and user['password'] == hash_password(password):
            return True
    return False

# Function to check if username exists
def username_exists(username):
    credentials = load_credentials()
    for user in credentials['users']:
        if user['username'] == username:
            return True
    return False

# Function to add a new user
def add_user(username, password, email):
    credentials = load_credentials()
    credentials['users'].append({
        'username': username,
        'password': hash_password(password),
        'email': email
    })
    save_credentials(credentials)

# Streamlit app
st.title('Hundred Win Predictor')

# Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Dummy variable to force rerun
if 'force_rerun' not in st.session_state:
    st.session_state.force_rerun = False

# Tabs for login and signup
if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        st.header("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()  # Force a rerun of the app to reflect the login state immediately
            else:
                st.error("Invalid username or password")

    with tab2:
        st.header("Signup")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        email = st.text_input("Email")
        if st.button("Signup"):
            if username_exists(new_username):
                st.error("Username already exists")
            else:
                add_user(new_username, new_password, email)
                st.success("Signup successful! You can now login.")
else:
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update(logged_in=False, username=None, force_rerun=not st.session_state.force_rerun))
    st.success(f"Welcome, {st.session_state.username}!")

    
    # Declaring the teams
    teams = [
        'Oval Invincibles',
        'London Spirit',
        'Southern Brave',
        'Welsh Fire',
        'Birmingham Phoenix',
        'Northern Superchargers',
        'Trent Rockets',
        'Manchester Originals'
    ]

    # Declaring the cities where the matches are going to take place
    cities = ['London', 'Birmingham', 'Nottingham', 'Leeds', 'Manchester', 'Cardiff', 'Southampton']
    import sklearn
    print(sklearn.__version__)
    # Loading our machine learning model from a saved pickle file
    pipe = pickle.load(open('pipe.pkl', 'rb'))  # remember all folders including pipe.pkl, notebook, datasets exist in the same directory
    score1 = pickle.load(open('score.pkl', 'rb'))

    # Prediction type selection
    prediction_type = st.selectbox('Select Prediction Type', ['Win Prediction', 'Score Prediction'])

    if prediction_type == 'Win Prediction':
        # Setting up the layout with two columns
        col1, col2 = st.columns(2)

        # Creating a dropdown selector for the batting team
        with col1:
            battingteam = st.selectbox('Select the batting team', sorted(teams))

        # Creating a dropdown selector for the bowling team
        with col2:
            bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

        # Creating a dropdown selector for the city where the match is being played
        city = st.selectbox('Select the city where the match is being played', sorted(cities))

        # Creating a numeric input for the target score using number_input method in streamlit
        target = int(st.number_input('Target', step=1))

        # Setting up the layout with three columns
        col3, col4, col5 = st.columns(3)

        # Creating a numeric input for the current score
        with col3:
            score = int(st.number_input('Score', step=1))

        # Creating a numeric input for the number of overs completed
        with col4:
            overs = int(st.number_input('Overs Completed', step=1))

        # Creating a numeric input for the number of wickets fallen
        with col5:
            wickets = int(st.number_input('Wickets Fallen', step=1))

        # Checking for different match results based on the input provided
        if score > target:
            st.write(battingteam, "won the match")

        elif score == target - 1 and overs == 20:
            st.write("Match Drawn")

        elif wickets == 10 and score < target - 1:
            st.write(bowlingteam, 'Won the match')

        elif wickets == 10 and score == target - 1:
            st.write('Match tied')

        elif battingteam == bowlingteam:
            st.write('To proceed, please select different teams because no match can be played between the same teams')

        else:
            # Checking if the input values are valid or not
            if target >= 0 and target <= 300 and overs >= 0 and overs <= 20 and wickets <= 10 and wickets >= 0 and score >= 0:
                try:
                    if st.button('Predict Probability'):
                        # Calculating the number of runs left for the batting team to win
                        runs_left = target - score

                        # Calculating the number of balls left
                        balls_left = 100 - (overs * 5)

                        # Calculating the number of wickets left for the batting team
                        wickets_left = 10 - wickets

                        # Calculating the current Run-Rate of the batting team
                        currentrunrate = score / overs

                        # Calculating the Required Run-Rate for the batting team to win
                        requiredrunrate = (runs_left * 5) / balls_left

                        # Creating a pandas DataFrame containing the user inputs
                        input_df = pd.DataFrame(
                            {'batting_team': [battingteam],
                             'bowling_team': [bowlingteam],
                             'city ': [city],
                             'runs_left': [runs_left],
                             'balls_left': [balls_left],
                             'wickets_left': [wickets_left],
                             'total-runs_x': [target],
                             'cur_run_rate': [currentrunrate],
                             'req_run_rate': [requiredrunrate]})
                        # Loading the trained machine learning pipeline to make the prediction
                        result = pipe.predict_proba(input_df)

                        # Extracting the likelihood of loss and win
                        lossprob = result[0][0]
                        winprob = result[0][1]

                        # Displaying the predicted likelihood of winning and losing in percentage
                        st.header(battingteam + "- " + str(round(winprob * 100)) + "%")
                        st.header(bowlingteam + "- " + str(round(lossprob * 100)) + "%")

                # Catching ZeroDivisionError
                except ZeroDivisionError:
                    st.error("Please fill all the details")

            # Displaying an error message if the input is incorrect
            else:
                st.error('There is something wrong with the input, please fill the correct details')

    elif prediction_type == 'Score Prediction':
        # Setting up the layout with two columns
        col1, col2 = st.columns(2)

        # Creating a dropdown selector for the batting team
        with col1:
            battingteam = st.selectbox('Select the batting team', sorted(teams))

        # Creating a dropdown selector for the bowling team
        with col2:
            bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

        # Creating a dropdown selector for the city where the match is being played
        city = st.selectbox('Select the city where the match is being played', sorted(cities))

        # Setting up the layout with three columns
        col3, col4, col5 = st.columns(3)

        # Creating a numeric input for the current score
        with col3:
            score = int(st.number_input('Score', step=1))

        # Creating a numeric input for the number of overs completed
        with col4:
            overs = int(st.number_input('Overs Completed', step=1))

        # Creating a numeric input for the number of wickets fallen
        with col5:
            wickets = int(st.number_input('Wickets Fallen', step=1))

        # Checking if the input values are valid or not
        if overs >= 0 and overs <= 20 and wickets <= 10 and wickets >= 0 and score >= 0:
            if battingteam == bowlingteam:
                st.write('To proceed, please select different teams because no match can be played between the same teams')
            else:
                try:
                    if st.button('Predict Score'):
                        # Calculating the number of balls left
                        balls_left = 100 - (overs * 5)

                        # Calculating the number of wickets left for the batting team
                        wickets_left = 10 - wickets

                        # Calculating the current Run-Rate of the batting team
                        currentrunrate = score / overs if overs > 0 else 0

                        # Creating a pandas DataFrame containing the user inputs
                        input_df = pd.DataFrame(
                            {'batting_team': [battingteam],
                             'bowling_team': [bowlingteam],
                             'city ': [city],
                             'balls_left': [balls_left],
                             'cur_run_rate': [currentrunrate],
                             'wickets_left': [wickets_left],
                             'over': [overs]
                             })

                        # Loading the trained machine learning pipeline to make the prediction
                        predicted_score = score1.predict(input_df)[0]
                        st.header(f"Predicted Final Score: {round(predicted_score * 100)}")

                        # Displaying the predicted score
                        

                # Catching ZeroDivisionError
                except ZeroDivisionError:
                    st.error("Please fill all the details")

        # Displaying an error message if the input is incorrect
        else:
            st.error('There is something wrong with the input, please fill the correct details')