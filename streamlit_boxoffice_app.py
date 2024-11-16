import numpy as np
import pandas as pd
import streamlit as st
import pickle
import os
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae

# Function to train and save the model and scaler
def train_and_save_model():
    # Read and preprocess the data
    df = pd.read_csv('boxoffice.csv', encoding='latin-1')
    df.drop(['world_revenue', 'opening_revenue'], axis=1, inplace=True)
    
    # Fill missing values
    df['MPAA'] = df['MPAA'].fillna(df['MPAA'].mode()[0])
    df['genres'] = df['genres'].fillna(df['genres'].mode()[0])
    df.dropna(inplace=True)

    df['domestic_revenue'] = df['domestic_revenue'].astype(str).str[1:]
    for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # One-hot encode the genres column
    vectorizer = CountVectorizer()
    vectorizer.fit(df['genres'])
    features = vectorizer.transform(df['genres']).toarray()

    genres = vectorizer.get_feature_names_out()
    for i, name in enumerate(genres):
        df[name] = features[:, i]
    df.drop('genres', axis=1, inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    df['MPAA'] = le.fit_transform(df['MPAA'])
    df['distributor'] = le.fit_transform(df['distributor'])

    # Log transformation on selected features
    features = ['domestic_revenue', 'opening_theaters', 'release_days']
    for col in features:
        df[col] = df[col].apply(lambda x: np.log10(x))

    # Split the data into train and validation sets
    X = df.drop(['title', 'domestic_revenue'], axis=1)
    y = df['domestic_revenue'].values
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.1, random_state=22)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train the model
    model = XGBRegressor()
    model.fit(X_train, Y_train)

    # Save the scaler and model
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model, scaler

# Check if scaler and model files exist
def load_model_and_scaler():
    if os.path.exists('scaler.pkl') and os.path.exists('model.pkl'):
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, scaler
    else:
        st.warning("Model and scaler not found, training the model...")
        model, scaler = train_and_save_model()
        return model, scaler

# Streamlit UI
st.title("Box Office Revenue Prediction")

# Load or train the model
model, scaler = load_model_and_scaler()

if model and scaler:
    st.sidebar.header("Input Movie Details")
    title = st.sidebar.text_input("Movie Title")
    distributor = st.sidebar.selectbox("Distributor", ['Warner Bros.', 'Disney', 'Sony', 'Paramount', 'Universal'])
    MPAA = st.sidebar.selectbox("MPAA Rating", ['R', 'PG-13', 'PG', 'G', 'NC-17'])
    genres = st.sidebar.text_input("Genres (comma separated)")
    opening_theaters = st.sidebar.number_input("Opening Theaters", min_value=1)
    release_days = st.sidebar.number_input("Release Days", min_value=1)

    # Convert the genres input into the required one-hot encoding format
    genre_list = genres.split(',')
    feature_names = model.get_booster().feature_names  # Get feature names from the trained model
    genre_features = [0] * len(feature_names)  # Initialize the genre features

    # Update genre features based on user input
    for genre in genre_list:
        genre = genre.strip().lower()
        if genre in feature_names:
            genre_features[feature_names.index(genre)] = 1

    # Prepare input data
    input_data = np.array([[
        opening_theaters,
        release_days,
        *genre_features
    ]])

    # Scale the input data
    input_data = scaler.transform(input_data)

    # Predict the revenue
    prediction = model.predict(input_data)

    st.subheader(f"Predicted Domestic Revenue: ${prediction[0]:,.2f}")

# Display the raw data
st.subheader("Dataset Preview")
df = pd.read_csv('boxoffice.csv', encoding='latin-1')
st.dataframe(df.head())
