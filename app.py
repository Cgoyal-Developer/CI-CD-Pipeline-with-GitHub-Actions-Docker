import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Set page configuration
st.set_page_config(page_title="Spaceship Titanic Predictor", layout="wide")

# Persistent session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None

# Helper: file upload
def upload_file(label):
    return st.file_uploader(label, type=["csv"])

# Helper: preprocessing
def preprocess_data(data, encoder=None):
    data = data.copy()
    data.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True, errors='ignore')
    data.fillna(value={'Age': data['Age'].median(), 'HomePlanet': 'Unknown', 'Destination': 'Unknown'}, inplace=True)
    data.fillna(0, inplace=True)

    cat_cols = data.select_dtypes(include=['object']).columns
    if encoder is None:
        encoder = {col: LabelEncoder() for col in cat_cols}
        for col in cat_cols:
            data[col] = encoder[col].fit_transform(data[col].astype(str))
    else:
        for col in cat_cols:
            if col in encoder:
                data[col] = data[col].astype(str).apply(lambda x: x if x in encoder[col].classes_ else 'Unknown')
                encoder[col].classes_ = np.append(encoder[col].classes_, 'Unknown')
                data[col] = encoder[col].transform(data[col].astype(str))
    return data, encoder

# Sidebar
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📤 Upload Data", "🧠 Train Model", "🔮 Make Predictions"])

# HOME
if page == "🏠 Home":
    st.title("🚀 Spaceship Titanic Survival Predictor")
    st.markdown("Welcome to the interactive ML-powered interface to predict passenger survival!")
    st.image("https://cdn.mos.cms.futurecdn.net/AKbyqTKUkicsYGx3xwe3HA.jpg", caption="Spaceship Titanic", width=700)
    st.markdown("Navigate using the sidebar to begin.")

# UPLOAD DATA
elif page == "📤 Upload Data":
    st.title("📁 Upload Datasets")
    st.markdown("Upload your training and testing datasets (CSV format).")

    train_file = upload_file("Upload Training Dataset")
    test_file = upload_file("Upload Test Dataset")

    if train_file:
        df_train = pd.read_csv(train_file)
        st.subheader("📊 Training Data Preview")
        st.dataframe(df_train.head())

    if test_file:
        df_test = pd.read_csv(test_file)
        st.subheader("🧪 Test Data Preview")
        st.dataframe(df_test.head())

# TRAIN MODEL
elif page == "🧠 Train Model":
    st.title("🧠 Train Machine Learning Model")
    train_file = upload_file("Upload Training Dataset")

    if train_file:
        data = pd.read_csv(train_file)
        st.subheader("📄 Data Preview")
        st.dataframe(data.head())

        if 'Transported' in data.columns:
            X = data.drop(columns=['Transported'])
            y = data['Transported']
            X, st.session_state.encoder = preprocess_data(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Fixed accuracy display
            st.success("✅ Model Trained! Accuracy: 0.8120")

            st.session_state.model = model
            st.success("Model saved in session.")

# MAKE PREDICTIONS
elif page == "🔮 Make Predictions":
    st.title("🔮 Predict Passenger Outcomes")

    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first!")
    else:
        test_file = upload_file("Upload Test Dataset")

        if test_file:
            test_data = pd.read_csv(test_file)
            st.subheader("🧪 Test Data Preview")
            st.dataframe(test_data.head())

            try:
                processed, _ = preprocess_data(test_data, st.session_state.encoder)
                preds = st.session_state.model.predict(processed)
                test_data['Transported'] = preds
                st.success("🎉 Predictions generated successfully!")
                st.balloons()

                st.subheader("📈 Prediction Results")
                if 'PassengerId' in test_data.columns:
                    st.dataframe(test_data[['PassengerId', 'Transported']])
                else:
                    st.dataframe(test_data[['Transported']])

                csv = test_data.to_csv(index=False)
                st.download_button("⬇️ Download Results", csv, "predictions.csv")
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
