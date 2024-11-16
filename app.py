# app.py
import streamlit as st
import pandas as pd
from model import train_model, predict

# Train the model (this is a one-time operation, then save the model)
model, scaler, le_distributor, le_mpaa, vectorizer = train_model()


# Streamlit User Interface
st.title('Box Office Revenue Prediction')

# Input fields for the user to enter their data
st.sidebar.header("Input Details")

# Input fields for user to provide movie details
opening_theaters = st.sidebar.number_input("Opening Theaters", min_value=1, value=1000)
release_days = st.sidebar.number_input("Release Days", min_value=1, value=30)
distributor = st.sidebar.selectbox("Distributor", ["Universal", "Warner Bros", "Paramount", "Sony", "Disney"])
mpaa = st.sidebar.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R", "NC-17"])
genres = st.sidebar.text_input("Genres", "Action, Comedy, Adventure")

# Submit button to make prediction
if st.sidebar.button("Predict Domestic Revenue"):
    # Prepare the user input as a dictionary
    user_input = {
        'opening_theaters': opening_theaters,
        'release_days': release_days,
        'distributor': distributor,
        'MPAA': mpaa,
        'genres': genres
    }
    
    # Get the prediction
    prediction = predict(model, scaler, le_distributor,le_mpaa, vectorizer, user_input)
    
    st.write(f"The predicted domestic revenue is: ${prediction:,.2f}")

