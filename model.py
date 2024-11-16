import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.feature_extraction.text import CountVectorizer

def train_model():
    # Load and preprocess the data
    df = pd.read_csv('boxoffice.csv', encoding='latin-1')

    # Drop unwanted columns
    to_remove = ['world_revenue', 'opening_revenue']
    df.drop(to_remove, axis=1, inplace=True)

    df.drop('budget', axis=1, inplace=True)
    for col in ['MPAA', 'genres']:
        df[col] = df[col].fillna(df[col].mode()[0])

    df.dropna(inplace=True)

    # Clean the 'domestic_revenue' column and convert to numeric
    df['domestic_revenue'] = df['domestic_revenue'].astype(str).str[1:]

    for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert genres to numerical features using CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(df['genres'])
    features = vectorizer.transform(df['genres']).toarray()

    genres = vectorizer.get_feature_names_out()
    for i, name in enumerate(genres):
        df[name] = features[:, i]

    df.drop('genres', axis=1, inplace=True)

    # Label encoding for distributor and MPAA columns
    le_distributor = LabelEncoder()
    le_mpaa = LabelEncoder()

    # Fit the encoders on all possible labels
    df['distributor'] = le_distributor.fit_transform(df['distributor'])
    df['MPAA'] = le_mpaa.fit_transform(df['MPAA'])

    # Feature and target separation
    features = df.drop(['title', 'domestic_revenue'], axis=1)
    target = df['domestic_revenue'].values

    # Train-test split
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train the model using XGBoost
    model = XGBRegressor()
    model.fit(X_train, Y_train)

    # Evaluate the model
    train_preds = model.predict(X_train)
    print('Training Error : ', mae(Y_train, train_preds))

    val_preds = model.predict(X_val)
    print('Validation Error : ', mae(Y_val, val_preds))

    # Return trained objects
    return model, scaler, le_distributor, le_mpaa, vectorizer



def predict(model, scaler, le_distributor, le_mpaa, vectorizer, user_input):
    # Convert genres to a vectorized form
    genres = vectorizer.transform([user_input['genres']]).toarray()
    
    # Handle unseen labels for distributor and MPAA by using the LabelEncoder's `classes_` attribute
    distributor = user_input['distributor']
    mpaa = user_input['MPAA']
    
    # Transform the distributor and MPAA values, and handle unseen labels
    if distributor in le_distributor.classes_:
        distributor = le_distributor.transform([distributor])[0]
    else:
        distributor = -1  # Assign a default value if unseen label

    if mpaa in le_mpaa.classes_:
        mpaa = le_mpaa.transform([mpaa])[0]
    else:
        mpaa = -1  # Assign a default value if unseen label
    
    # Prepare features for prediction
    features = np.array([
        user_input['opening_theaters'],
        user_input['release_days'],
        distributor,
        mpaa
    ])

    # Combine the genre features
    features = np.concatenate([features, genres.flatten()])
    
    # Reshape the input for prediction
    features = scaler.transform([features])

    # Make the prediction
    prediction = model.predict(features)

    return prediction[0]
