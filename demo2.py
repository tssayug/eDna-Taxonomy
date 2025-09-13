# ==================================================================================================
# eDNA Biodiversity Analysis Tool (v1.2 - Stratify Fix)
#
# Author: 
# Date: September 13, 2025
#
# CORRECTIONS:
# 1. Expanded the embedded sample dataset to ensure every taxonomic family has at least 2 members.
#    This fixes the `ValueError` during train_test_split with stratification.
# 2. Kept previous fixes (embedded data, use_container_width).
# ==================================================================================================

# Core Libraries
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import io
import requests
from contextlib import closing

# Data Visualization
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning - Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Other Utilities
from urllib.error import URLError
import base64
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="eDNA Biodiversity Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com',
        'Report a bug': "https://www.example.com",
        'About': """
        ## eDNA Biodiversity Explorer
        
        **Version 1.2 (Stratify Fix)**
        
        This app  provides a full-stack solution
        for analyzing environmental DNA (eDNA) datasets to assess biodiversity.
        """
    }
)

# --- Database Management (SQLite) ---

DB_NAME = "edna_data.db"

def init_db():
    """Initializes the SQLite database and creates tables if they don't exist."""
    with closing(sqlite3.connect(DB_NAME)) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    project_name TEXT PRIMARY KEY,
                    file_name TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Using TEXT for dataframes to store them as CSV strings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    project_name TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    FOREIGN KEY (project_name) REFERENCES projects(project_name)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    project_name TEXT PRIMARY KEY,
                    predictions_data TEXT NOT NULL,
                    FOREIGN KEY (project_name) REFERENCES projects(project_name)
                )
            ''')
        conn.commit()

def get_project_list():
    """Retrieves the list of saved project names from the database."""
    with closing(sqlite3.connect(DB_NAME)) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT project_name FROM projects ORDER BY upload_date DESC")
            projects = cursor.fetchall()
            return [proj[0] for proj in projects]

def save_data(project_name, file_name, df):
    """Saves project info and its dataframe to the database."""
    if not project_name.strip():
        st.error("Project name cannot be empty.")
        return False
    try:
        csv_string = df.to_csv(index=False)
        with closing(sqlite3.connect(DB_NAME)) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute("INSERT OR REPLACE INTO projects (project_name, file_name) VALUES (?, ?)", (project_name, file_name))
                cursor.execute("INSERT OR REPLACE INTO datasets (project_name, data) VALUES (?, ?)", (project_name, csv_string))
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Database Error: Failed to save project '{project_name}'. Reason: {e}")
        return False

def save_predictions(project_name, df_predictions):
    """Saves predictions for a project to the database."""
    try:
        csv_string = df_predictions.to_csv(index=False)
        with closing(sqlite3.connect(DB_NAME)) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute("INSERT OR REPLACE INTO predictions (project_name, predictions_data) VALUES (?, ?)", (project_name, csv_string))
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Database Error: Failed to save predictions for '{project_name}'. Reason: {e}")
        return False

def load_data(project_name, table='datasets'):
    """Loads a dataframe or predictions for a given project name."""
    try:
        with closing(sqlite3.connect(DB_NAME)) as conn:
            with closing(conn.cursor()) as cursor:
                if table == 'datasets':
                    cursor.execute("SELECT data FROM datasets WHERE project_name = ?", (project_name,))
                else:
                    cursor.execute("SELECT predictions_data FROM predictions WHERE project_name = ?", (project_name,))
                
                result = cursor.fetchone()
                if result:
                    csv_string = result[0]
                    df = pd.read_csv(io.StringIO(csv_string))
                    return df
                else:
                    st.warning(f"No data found for project '{project_name}' in table '{table}'.")
                    return None
    except Exception as e:
        st.error(f"Database Error: Failed to load data for '{project_name}'. Reason: {e}")
        return None

# Initialize the database on first run
init_db()


# --- Sample Data Loading (FIXED & EXPANDED) ---
@st.cache_data
def load_sample_data():
    """
    Loads a sample eDNA dataset from an embedded CSV string.
    The dataset is now expanded to ensure all classes have at least 2 members for stratification.
    """
    csv_data = """
kingdom,phylum,class,order,family,genus,species_name,dna_sequence
Animalia,Chordata,Actinopterygii,Cypriniformes,Cyprinidae,Rutilus,Rutilus rutilus,AGTCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
Animalia,Chordata,Actinopterygii,Cypriniformes,Cyprinidae,Rutilus,Rutilus rutilus,AGTCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
Animalia,Chordata,Actinopterygii,Cypriniformes,Cyprinidae,Abramis,Abramis brama,CGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
Animalia,Chordata,Actinopterygii,Cypriniformes,Cyprinidae,Abramis,Abramis brama,CGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
Animalia,Chordata,Actinopterygii,Perciformes,Percidae,Perca,Perca fluviatilis,GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
Animalia,Chordata,Actinopterygii,Perciformes,Percidae,Perca,Perca fluviatilis,GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
Animalia,Chordata,Actinopterygii,Esociformes,Esocidae,Esox,Esox lucius,TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGT
Animalia,Chordata,Actinopterygii,Esociformes,Esocidae,Esox,Esox lucius,TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGT
Animalia,Chordata,Actinopterygii,Salmoniformes,Salmonidae,Salmo,Salmo trutta,AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
Animalia,Chordata,Actinopterygii,Salmoniformes,Salmonidae,Salmo,Salmo trutta,AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
Animalia,Chordata,Actinopterygii,Cypriniformes,Cyprinidae,Gobio,Gobio gobio,GTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
Animalia,Chordata,Actinopterygii,Cypriniformes,Cyprinidae,Gobio,Gobio gobio,GTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
Animalia,Chordata,Actinopterygii,Perciformes,Percidae,Sander,Sander lucioperca,CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
Animalia,Chordata,Actinopterygii,Perciformes,Percidae,Sander,Sander lucioperca,CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
Animalia,Chordata,Actinopterygii,Cypriniformes,Cyprinidae,Alburnus,Alburnus alburnus,TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
Animalia,Chordata,Actinopterygii,Cypriniformes,Cyprinidae,Alburnus,Alburnus alburnus,TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
Animalia,Chordata,Actinopterygii,Anguilliformes,Anguillidae,Anguilla,Anguilla anguilla,GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
Animalia,Chordata,Actinopterygii,Anguilliformes,Anguillidae,Anguilla,Anguilla anguilla,GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
Animalia,Chordata,Actinopterygii,Cypriniformes,Cobitidae,Cobitis,Cobitis taenia,AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
Animalia,Chordata,Actinopterygii,Cypriniformes,Cobitidae,Cobitis,Cobitis taenia,AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
Animalia,Chordata,Actinopterygii,Cypriniformes,Cyprinidae,Leuciscus,Leuciscus idus,GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
Animalia,Chordata,Actinopterygii,Cypriniformes,Cyprinidae,Leuciscus,Leuciscus idus,GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG
Animalia,Chordata,Actinopterygii,Perciformes,Gobiidae,Neogobius,Neogobius melanostomus,CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
Animalia,Chordata,Actinopterygii,Perciformes,Gobiidae,Neogobius,Neogobius melanostomus,CTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
"""
    try:
        # Use io.StringIO to read the string data as if it were a file
        df = pd.read_csv(io.StringIO(csv_data))
        # Basic cleaning for consistency
        df = df.dropna(subset=['dna_sequence', 'family', 'genus', 'species_name'])
        df['species_name'] = df['species_name'].str.replace('_', ' ')
        return df
    except Exception as e:
        st.error(f"Failed to load embedded sample data. Error: {e}")
        return pd.DataFrame()


# --- AI/ML Pipeline ---

def dna_to_kmers(sequence, k=4):
    """Converts a DNA sequence into a string of k-mers."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

@st.cache_resource
def train_taxonomy_model(df_train):
    """Trains a Random Forest model for taxonomy classification based on k-mers."""
    
    st.info("Starting AI Model Training...")
    progress_bar = st.progress(0, text="Preparing data...")
    
    # 1. Feature Engineering: Convert DNA to k-mers
    df_train['kmers'] = df_train['dna_sequence'].apply(dna_to_kmers)
    
    # 2. Vectorization
    time.sleep(1)
    progress_bar.progress(20, text="Vectorizing DNA sequences (k-mers)...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df_train['kmers'])
    
    # 3. Label Encoding
    # We will predict the 'family' as our target taxonomy level
    target_column = 'family'
    le = LabelEncoder()
    y = le.fit_transform(df_train[target_column])
    
    # 4. Train-Test Split
    time.sleep(1)
    progress_bar.progress(40, text="Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 5. Model Training
    progress_bar.progress(60, text="Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 6. Evaluation
    time.sleep(1)
    progress_bar.progress(80, text="Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    progress_bar.progress(100, text="Training complete!")
    time.sleep(1)
    progress_bar.empty()
    
    # Store results in a dictionary
    model_artifacts = {
        'model': model,
        'vectorizer': vectorizer,
        'label_encoder': le,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'target_classes': le.classes_
    }
    
    return model_artifacts

def predict_taxonomy(df_predict, model_artifacts):
    """Uses a trained model to predict taxonomy for new eDNA sequences."""
    # Preprocessing
    df_predict['kmers'] = df_predict['dna_sequence'].apply(dna_to_kmers)
    
    # Vectorization
    X_new = model_artifacts['vectorizer'].transform(df_predict['kmers'])
    
    # Prediction
    predictions_encoded = model_artifacts['model'].predict(X_new)
    prediction_probabilities = model_artifacts['model'].predict_proba(X_new)
    
    # Decoding predictions
    predicted_labels = model_artifacts['label_encoder'].inverse_transform(predictions_encoded)
    confidence_scores = np.max(prediction_probabilities, axis=1)
    
    # Add to dataframe
    df_predict['predicted_family'] = predicted_labels
    df_predict['prediction_confidence'] = confidence_scores
    
    return df_predict.drop(columns=['kmers'])


# --- UI Helper Functions ---

def get_df_download_link(df, filename="results.csv", text="Download CSV"):
    """Generates a link to download a pandas DataFrame as a CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

def parse_uploaded_file(uploaded_file):
    """Parses uploaded CSV, FASTA, or JSON files into a DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.fasta', '.fa')):
            sequences = []
            headers = []
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            header = None
            seq = ""
            for line in stringio:
                line = line.strip()
                if line.startswith(">"):
                    if header:
                        headers.append(header)
                        sequences.append(seq)
                    header = line[1:]
                    seq = ""
                else:
                    seq += line
            if header: # Append the last sequence
                headers.append(header)
                sequences.append(seq)
            return pd.DataFrame({'identifier': headers, 'dna_sequence': sequences})
        elif uploaded_file.name.endswith('.json'):
            return pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, FASTA, or JSON.")
            return None
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None


# --- Main Application UI ---
# (The rest of the file is unchanged)

# --- Sidebar Navigation ---
st.sidebar.title("🧬 eDNA Biodiversity Explorer")
st.sidebar.markdown("---")

# Session State Initialization
if 'current_project' not in st.session_state:
    st.session_state.current_project = None
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
if 'df_with_predictions' not in st.session_state:
    st.session_state.df_with_predictions = None

# Project selection
st.sidebar.header("📂 Manage Projects")
project_list = get_project_list()
# Add a default option for the sample data
project_options = ["-- Use Sample Dataset --"] + project_list

# Check if the current project still exists in the list
if st.session_state.current_project and st.session_state.current_project not in project_options:
    st.session_state.current_project = None
    st.session_state.df_loaded = None
    st.session_state.df_with_predictions = None

selected_project = st.sidebar.selectbox(
    "Load Existing Project or Use Sample",
    project_options,
    index=0, # Default to sample data
    key="project_selector",
)

if st.sidebar.button("Load Project", use_container_width=True):
    if selected_project != "-- Use Sample Dataset --":
        st.session_state.current_project = selected_project
        st.session_state.df_loaded = load_data(selected_project, 'datasets')
        # Also try to load predictions if they exist
        st.session_state.df_with_predictions = load_data(selected_project, 'predictions')
        st.sidebar.success(f"Loaded project: {selected_project}")
    else:
        st.session_state.current_project = "Sample Dataset"
        st.session_state.df_loaded = load_sample_data()
        st.session_state.df_with_predictions = None # Reset predictions for sample
        st.sidebar.info("Switched to the sample dataset.")

# Set active DataFrame based on selection
if st.session_state.df_loaded is None:
    active_df = load_sample_data()
    if not active_df.empty:
        st.session_state.current_project = "Sample Dataset"
else:
    active_df = st.session_state.df_loaded

if st.session_state.current_project:
    st.sidebar.markdown(f"**Active Project:** `{st.session_state.current_project}`")

st.sidebar.markdown("---")

# Main Page Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📤 Upload & Manage Data",
    "📊 Data Visualization",
    "🤖 AI/ML Taxonomy Prediction",
    "💡 Insights & Recommendations"
])

# --- Page Content ---

if page == "🏠 Home":
    st.title("🏠 Welcome to the eDNA Biodiversity Explorer")
    st.markdown("---")
    st.image("https://images.unsplash.com/photo-1637929476734-bd7f5f78e40a?q=80&w=1632&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
             caption="Environmental DNA (eDNA) is revolutionizing how we monitor biodiversity. Photo by USGS on Unsplash.",
             use_container_width=True)
    
    st.header("The Challenge: Unlocking Biodiversity from Genetic Traces")
    st.markdown("""
        Environmental DNA (eDNA) — genetic material collected directly from environmental samples like soil, water, or air — offers a non-invasive, powerful way to detect species. However, raw eDNA datasets are vast and complex, consisting of millions of short DNA sequences. 
        
        The critical challenge is to accurately identify which species these sequences belong to (taxonomy) and to translate this data into meaningful biodiversity metrics. This tool is designed to bridge that gap.
    """)

    st.header("How This App Works")
    st.markdown("""
        This interactive dashboard provides a complete workflow for eDNA analysis, from data upload to actionable insights, all within a single platform.
        
        1.  **📤 Upload & Manage Data:** Start by uploading your own eDNA dataset (in CSV, FASTA, or JSON format) or use the built-in sample fish dataset. All your uploaded datasets are saved as 'projects' in a local database for you to reload later.
        
        2.  **🤖 AI/ML Taxonomy Prediction:** Our core feature. The app uses a pre-trained Machine Learning model (Random Forest) to classify DNA sequences and predict their taxonomic family. You can run this prediction on your own uploaded data.
        
        3.  **📊 Data Visualization:** Interactively explore your data. The app automatically generates charts for species richness, abundance, and a hierarchical sunburst plot to visualize the tree of life within your sample.
        
        4.  **💡 Insights & Recommendations:** Get a summarized biodiversity assessment based on the analysis, highlighting key findings and potential ecological implications.
    """)

    st.info("""
        **Getting Started:**
        - **To use the sample data:** Navigate to any section. The app loads a fish eDNA dataset by default.
        - **To use your own data:** Go to the **'📤 Upload & Manage Data'** section to upload a file and create a new project. Then, load it from the sidebar.
    """)

elif page == "📤 Upload & Manage Data":
    st.title("📤 Upload & Manage eDNA Data")
    st.markdown("---")
    
    st.header("Upload New Dataset")
    st.markdown("Create a new project by uploading your eDNA data. Supported formats: CSV, FASTA, JSON.")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click to browse",
        type=['csv', 'fasta', 'fa', 'json']
    )
    
    if uploaded_file is not None:
        df_upload = parse_uploaded_file(uploaded_file)
        
        if df_upload is not None:
            st.success(f"Successfully parsed **{uploaded_file.name}**. Found {len(df_upload)} records.")
            st.dataframe(df_upload.head())
            
            # Check for required 'dna_sequence' column
            if 'dna_sequence' not in df_upload.columns:
                st.error("The uploaded file must contain a column named 'dna_sequence'.")
            else:
                project_name = st.text_input("Enter a unique name for this project:")
                if st.button("Save Project"):
                    if save_data(project_name, uploaded_file.name, df_upload):
                        st.success(f"Project '{project_name}' saved successfully!")
                        st.info("You can now load this project from the sidebar to analyze it.")
                        # Force a rerun to update the sidebar project list
                        st.rerun()

    st.markdown("---")
    st.header("Current Projects")
    st.markdown("Here is a list of all projects you have saved to the local database.")
    
    projects = get_project_list()
    if not projects:
        st.warning("No projects saved yet. Upload a dataset to get started.")
    else:
        st.table(pd.DataFrame({"Saved Project Names": projects}))


elif page == "🤖 AI/ML Taxonomy Prediction":
    st.title("🤖 AI/ML Taxonomy Prediction")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1637929476734-bd7f5f78e40a?q=80&w=1632&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", 
                 caption="Machine learning models can learn patterns in DNA to predict taxonomy.",
                 use_container_width=True)
        st.subheader("Model Details")
        st.markdown("""
        - **Model:** Random Forest Classifier
        - **Feature Engineering:** 4-mer DNA sequence tokenization
        - **Target:** Taxonomic `family`
        - **Training Data:** The built-in sample fish eDNA dataset.
        """)

    with col2:
        st.header("Train or Use the AI Model")
        st.markdown("The model is trained on the sample dataset to predict the taxonomic family from a DNA sequence.")

        # Train model (or load from cache)
        sample_df = load_sample_data()
        if not sample_df.empty:
            with st.spinner("Loading and preparing the AI model... This may take a moment on first run."):
                model_artifacts = train_taxonomy_model(sample_df)
            
            st.success("AI model is ready!")
            st.metric("Model Accuracy on Test Data", f"{model_artifacts['accuracy']:.2%}")
        else:
            st.error("Could not train model because the sample data failed to load.")

    st.markdown("---")
    st.header(f"Run Predictions on '{st.session_state.current_project}'")

    if not sample_df.empty:
        if st.session_state.current_project == "Sample Dataset":
            st.info("The model is already trained on the sample data. To see predictions on new data, please upload your own dataset first.")
        
        elif active_df is None:
            st.warning("Please upload and load a dataset first before running predictions.")
        
        elif 'dna_sequence' not in active_df.columns:
            st.error("The loaded dataset does not have a 'dna_sequence' column, which is required for prediction.")

        else:
            if st.button("🚀 Run Taxonomy Prediction", use_container_width=True):
                with st.spinner("Classifying sequences... This might take a few moments for large datasets."):
                    df_with_preds = predict_taxonomy(active_df.copy(), model_artifacts)
                    st.session_state.df_with_predictions = df_with_preds
                
                st.success("Prediction complete!")
                st.dataframe(st.session_state.df_with_predictions.head())
                
                # Save predictions
                if save_predictions(st.session_state.current_project, st.session_state.df_with_predictions):
                    st.info(f"Predictions saved for project '{st.session_state.current_project}'.")
                
                st.markdown(get_df_download_link(st.session_state.df_with_predictions,
                                                 f"{st.session_state.current_project}_predictions.csv",
                                                 "Download Predictions as CSV"), unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("Model Performance Deep Dive")
    st.markdown("This shows how the model performed on the held-out test portion of the sample dataset.")
    
    if not sample_df.empty:
        with st.expander("Show Classification Report"):
            report_df = pd.DataFrame(model_artifacts['classification_report']).transpose()
            st.dataframe(report_df)

        with st.expander("Show Confusion Matrix"):
            fig = go.Figure(data=go.Heatmap(
                z=model_artifacts['confusion_matrix'],
                x=model_artifacts['target_classes'],
                y=model_artifacts['target_classes'],
                colorscale='Blues'
            ))
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted Family',
                yaxis_title='Actual Family'
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Data Visualization":
    st.title("📊 Interactive Data Visualization")
    st.markdown("---")

    # Use the dataframe with predictions if available, otherwise use the base loaded dataframe
    df_to_visualize = st.session_state.df_with_predictions if st.session_state.df_with_predictions is not None else active_df

    if df_to_visualize is None or df_to_visualize.empty:
        st.warning("No data to visualize. Please upload a dataset or load the sample data.")
    else:
        st.header(f"Biodiversity Dashboard for: `{st.session_state.current_project}`")
        
        # Determine the primary column for species identification
        if 'species_name' in df_to_visualize.columns:
            species_col = 'species_name'
        elif 'identifier' in df_to_visualize.columns:
            species_col = 'identifier'
        else:
            st.warning("Could not find a clear species identifier column like 'species_name' or 'identifier'. Some charts may not work.")
            species_col = None

        # Determine taxonomy column - prefer predicted if available
        if 'predicted_family' in df_to_visualize.columns:
            family_col = 'predicted_family'
        elif 'family' in df_to_visualize.columns:
            family_col = 'family'
        else:
            family_col = None
            
        # --- Key Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Total DNA Sequences", len(df_to_visualize))
        if species_col:
            col2.metric("Species Richness (Unique Species)", df_to_visualize[species_col].nunique())
        if family_col:
            col3.metric("Taxonomic Families Identified", df_to_visualize[family_col].nunique())
        
        st.markdown("---")

        # --- Visualizations ---
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.subheader("Species Abundance Distribution")
            if species_col:
                species_counts = df_to_visualize[species_col].value_counts().reset_index()
                species_counts.columns = [species_col, 'count']
                fig_bar = px.bar(species_counts.head(20), x=species_col, y='count',
                                 title="Top 20 Most Abundant Species",
                                 labels={'count': 'Number of DNA Reads', species_col: 'Species Name'})
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("A 'species_name' or 'identifier' column is needed for this chart.")

        with viz_col2:
            st.subheader("Taxonomic Family Composition")
            if family_col:
                family_counts = df_to_visualize[family_col].value_counts()
                fig_pie = px.pie(family_counts, values=family_counts.values, names=family_counts.index,
                                 title="Proportion of Sequences by Family",
                                 hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("A 'family' or 'predicted_family' column is needed for this chart.")

        st.markdown("---")
        st.subheader("Hierarchical Taxonomy Tree (Sunburst Chart)")

        # Check for required columns for the sunburst chart
        taxonomy_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species_name']
        if all(col in df_to_visualize.columns for col in taxonomy_cols):
             # Fill NaNs with a placeholder to avoid errors in path
            df_sunburst = df_to_visualize[taxonomy_cols].fillna('Unknown')
            fig_sunburst = px.sunburst(df_sunburst, path=taxonomy_cols,
                                       title="Interactive Taxonomic Hierarchy")
            fig_sunburst.update_layout(margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig_sunburst, use_container_width=True)
        else:
            st.info("""
            To generate the taxonomy tree, the dataset must contain the following columns: 
            `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, `species_name`.
            The sample dataset includes these. For uploaded data, this chart may not be available.
            """)
        
        st.markdown("---")
        st.subheader("Genetic Sequence Similarity (PCA Clustering)")
        
        try:
            sample_df = load_sample_data()
            if not sample_df.empty:
                with st.spinner("Performing PCA on DNA sequences..."):
                    # Use the same vectorizer from the trained model
                    model_artifacts = train_taxonomy_model(sample_df)
                    vectorizer = model_artifacts['vectorizer']
                    
                    # We need to fit the vectorizer on the current data if it's different
                    df_to_visualize['kmers'] = df_to_visualize['dna_sequence'].apply(dna_to_kmers)
                    X_viz = vectorizer.fit_transform(df_to_visualize['kmers'])
                    
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_viz.toarray())
                    
                    df_to_visualize[['pca1', 'pca2']] = X_pca
                    
                    color_col = family_col if family_col else species_col
                    
                    if color_col:
                        fig_pca = px.scatter(df_to_visualize, x='pca1', y='pca2', color=color_col,
                                             title="2D PCA of DNA k-mer Signatures",
                                             hover_data=[species_col] if species_col else [],
                                             labels={'pca1': 'Principal Component 1', 'pca2': 'Principal Component 2'})
                        st.plotly_chart(fig_pca, use_container_width=True)
                    else:
                        st.info("A column for coloring (e.g., 'family' or 'species_name') is needed for the PCA plot.")
        except Exception as e:
            st.error(f"Could not generate PCA plot. An error occurred: {e}")


elif page == "💡 Insights & Recommendations":
    st.title("💡 Insights & Recommendations")
    st.markdown("---")
    
    st.header(f"Biodiversity Summary for: `{st.session_state.current_project}`")

    # Use the dataframe with predictions if available
    df_insights = st.session_state.df_with_predictions if st.session_state.df_with_predictions is not None else active_df

    if df_insights is None or df_insights.empty:
        st.warning("No data available to generate insights.")
    else:
        # Determine columns to use
        species_col = 'species_name' if 'species_name' in df_insights.columns else None
        family_col = 'predicted_family' if 'predicted_family' in df_insights.columns else ('family' if 'family' in df_insights.columns else None)

        # Generate Insights
        st.subheader("Key Biodiversity Indicators")
        
        # Insight 1: Richness
        if species_col:
            richness = df_insights[species_col].nunique()
            st.info(f"**Species Richness:** A total of **{richness}** unique species were identified in this sample.")
            if richness > 20:
                st.markdown("This indicates a relatively high level of species diversity.")
            elif richness > 5:
                st.markdown("This suggests a moderate level of species diversity.")
            else:
                st.markdown("This points to a low level of species diversity in the analyzed sample.")

        # Insight 2: Dominance
        if species_col:
            species_counts = df_insights[species_col].value_counts()
            most_abundant_species = species_counts.index[0]
            abundance_percentage = (species_counts.iloc[0] / len(df_insights)) * 100
            st.info(f"**Dominant Species:** The most frequently detected species is **'{most_abundant_species}'**, accounting for **{abundance_percentage:.2f}%** of the total DNA sequences.")
            if abundance_percentage > 50:
                st.warning("A high dominance by a single species can sometimes indicate an ecological imbalance. Further investigation into why this species thrives in this environment may be warranted.")

        # Insight 3: Taxonomic Composition
        if family_col:
            family_counts = df_insights[family_col].value_counts()
            most_abundant_family = family_counts.index[0]
            st.info(f"**Dominant Family:** At the family level, **'{most_abundant_family}'** is the most represented taxonomic group.")
            if st.session_state.current_project == "Sample Dataset" and most_abundant_family == 'Cyprinidae':
                 st.markdown("The Cyprinidae family (which includes carps and minnows) is common in freshwater environments worldwide and their strong presence is expected in many riverine eDNA samples.")

        # Insight 4: Prediction Confidence (if available)
        if 'prediction_confidence' in df_insights.columns:
            avg_confidence = df_insights['prediction_confidence'].mean()
            st.subheader("AI Model Confidence")
            st.success(f"**Average Prediction Confidence:** The AI model's average confidence score for its taxonomic predictions on this dataset is **{avg_confidence:.2%}**.")
            if avg_confidence < 0.7:
                 st.markdown("A lower average confidence may suggest the presence of novel species not well-represented in the model's training data, or it could point to lower quality DNA sequences.")
            else:
                 st.markdown("High confidence scores indicate that the DNA sequences found are a good match for known species in the model's database.")
        
        st.markdown("---")
        st.header("Recommendations")
        st.markdown("""
        - **Comparative Analysis:** To draw stronger conclusions, compare these results with samples from different locations, times, or environmental conditions.
        - **Ground Truthing:** Correlate eDNA findings with traditional survey methods (e.g., visual counts, netting) to validate the presence of key identified species.
        - **Further Research:** Investigate the ecological roles of the dominant species and the potential absence of expected species to better understand the health of the ecosystem.
        - **Model Retraining:** For ongoing monitoring projects, consider retraining the AI model with your own validated, locally-relevant species data to improve prediction accuracy for your specific ecosystem.
        """)
        
        st.markdown("---")
        st.header("Download Processed Data")
        st.markdown("Download the complete dataset with any generated predictions for your own offline analysis.")
        st.markdown(get_df_download_link(df_insights, f"{st.session_state.current_project}_full_results.csv"), unsafe_allow_html=True)


# ==================================================================================================
# README Section (as comments)
# (Content is the same, no changes needed here)
# ==================================================================================================

