"""
Training script for Plan Prediction Models using Sentence Transformers.

This script:
1. Loads and preprocesses plan data
2. Generates sentence embeddings
3. Trains classification models
4. Saves models for API use
"""

import pandas as pd
import numpy as np
import re
import joblib
import warnings
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = r'dataset/Iris Plan_Mapping Data_Top 100.xlsx'
MODELS_DIR = 'models'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def clean_text(text):
    """Standardize text data for matching."""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[\W_]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess_data():
    """Load and preprocess the data."""
    print("Loading data...")
    df_raw = pd.read_excel(DATA_FILE, sheet_name=0)
    df_clean = pd.read_excel(DATA_FILE, sheet_name=3)
    
    print(f"Raw data: {df_raw.shape}")
    print(f"Clean data: {df_clean.shape}")
    
    # Clean text
    df_raw['PAYER NAME_cleaned'] = df_raw['PAYER NAME'].apply(clean_text)
    df_raw['PLAN NAME_dirty_cleaned'] = df_raw['PLAN NAME'].apply(clean_text)
    df_clean['PLAN NAME_cleaned'] = df_clean['PLAN NAME'].apply(clean_text)
    
    # Create combined dirty name
    df_raw['combined_dirty_name'] = (
        df_raw['PAYER NAME_cleaned'] + ' ' + df_raw['PLAN NAME_dirty_cleaned']
    )
    
    return df_raw, df_clean


def fuzzy_match_plans(df_raw, df_clean):
    """Perform fuzzy matching to align dirty and clean plan names."""
    print("\nPerforming fuzzy matching...")
    
    matched_clean_names = []
    match_scores = []
    clean_plan_choices = df_clean['PLAN NAME_cleaned'].tolist()
    
    for index, row in df_raw.iterrows():
        dirty_name = row['combined_dirty_name']
        if dirty_name:
            best_match = process.extractOne(
                dirty_name, clean_plan_choices, scorer=fuzz.token_set_ratio
            )
            if best_match:
                matched_clean_names.append(best_match[0])
                match_scores.append(best_match[1])
            else:
                matched_clean_names.append(np.nan)
                match_scores.append(0)
        else:
            matched_clean_names.append(np.nan)
            match_scores.append(0)
        
        if (index + 1) % 5000 == 0:
            print(f"  Processed {index + 1}/{len(df_raw)} rows...")
    
    df_raw['matched_clean_plan_name_cleaned'] = matched_clean_names
    df_raw['match_score'] = match_scores
    
    # Merge to get original clean plan values
    df_clean_for_merge = df_clean[[
        'PLAN NAME', 'PLAN_TYPE', 'LINE_OF_ BUSINESS', 'PLAN NAME_cleaned'
    ]].copy()
    df_clean_for_merge.rename(columns={
        'PLAN NAME': 'clean_plan_name',
        'PLAN_TYPE': 'clean_plan_type',
        'LINE_OF_ BUSINESS': 'clean_LOB',
        'PLAN NAME_cleaned': 'matched_clean_plan_name_cleaned'
    }, inplace=True)
    
    df_merged = pd.merge(
        df_raw, df_clean_for_merge, 
        on='matched_clean_plan_name_cleaned', 
        how='left'
    )
    
    print(f"Match score stats:\n{df_merged['match_score'].describe()}")
    return df_merged


def generate_embeddings(df_merged):
    """Generate sentence embeddings for all plan names."""
    print("\nLoading sentence transformer model...")
    sentence_model = SentenceTransformer(MODEL_NAME)
    
    print("Generating embeddings...")
    sentences = df_merged['combined_dirty_name'].tolist()
    embeddings = sentence_model.encode(
        sentences,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, sentence_model


def encode_targets(df_merged):
    """Encode target variables."""
    print("\nEncoding target variables...")
    
    le_plan_name = LabelEncoder()
    le_plan_type = LabelEncoder()
    le_lob = LabelEncoder()
    
    df_merged['encoded_clean_plan_name'] = le_plan_name.fit_transform(
        df_merged['clean_plan_name']
    )
    df_merged['encoded_clean_plan_type'] = le_plan_type.fit_transform(
        df_merged['clean_plan_type']
    )
    df_merged['encoded_clean_LOB'] = le_lob.fit_transform(
        df_merged['clean_LOB']
    )
    
    print(f"Unique plan names: {len(le_plan_name.classes_)}")
    print(f"Unique plan types: {len(le_plan_type.classes_)}")
    print(f"Unique LOBs: {len(le_lob.classes_)}")
    
    return le_plan_name, le_plan_type, le_lob


def train_models(embeddings, df_merged):
    """Train classification models."""
    print("\nTraining models...")
    
    X = embeddings
    y_plan_name = df_merged['encoded_clean_plan_name']
    y_plan_type = df_merged['encoded_clean_plan_type']
    y_lob = df_merged['encoded_clean_LOB']
    
    # Split data
    X_train_pn, X_test_pn, y_train_pn, y_test_pn = train_test_split(
        X, y_plan_name, test_size=0.2, random_state=42
    )
    X_train_pt, X_test_pt, y_train_pt, y_test_pt = train_test_split(
        X, y_plan_type, test_size=0.2, random_state=42
    )
    X_train_lob, X_test_lob, y_train_lob, y_test_lob = train_test_split(
        X, y_lob, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train_pn.shape[0]}")
    print(f"Test set size: {X_test_pn.shape[0]}")
    
    # Train models
    model_plan_name = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model_plan_name.fit(X_train_pn, y_train_pn)
    print("✓ Plan name model trained")
    
    model_plan_type = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model_plan_type.fit(X_train_pt, y_train_pt)
    print("✓ Plan type model trained")
    
    model_lob = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model_lob.fit(X_train_lob, y_train_lob)
    print("✓ LOB model trained")
    
    return (
        model_plan_name, model_plan_type, model_lob,
        (X_test_pn, y_test_pn), (X_test_pt, y_test_pt), (X_test_lob, y_test_lob)
    )


def evaluate_models(models, test_data, encoders):
    """Evaluate model performance."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    model_plan_name, model_plan_type, model_lob = models
    le_plan_name, le_plan_type, le_lob = encoders
    (X_test_pn, y_test_pn), (X_test_pt, y_test_pt), (X_test_lob, y_test_lob) = test_data
    
    # Evaluate plan name
    y_pred_pn = model_plan_name.predict(X_test_pn)
    print("\nPlan Name Accuracy:", 
          (y_pred_pn == y_test_pn).sum() / len(y_test_pn))
    
    # Evaluate plan type
    y_pred_pt = model_plan_type.predict(X_test_pt)
    print("Plan Type Accuracy:", 
          (y_pred_pt == y_test_pt).sum() / len(y_test_pt))
    labels_pt = np.unique(np.concatenate((y_test_pt, y_pred_pt)))
    target_names_pt = le_plan_type.inverse_transform(labels_pt)
    print("\nPlan Type Report:")
    print(classification_report(
        y_test_pt, y_pred_pt, labels=labels_pt, 
        target_names=target_names_pt, zero_division=0
    ))
    
    # Evaluate LOB
    y_pred_lob = model_lob.predict(X_test_lob)
    print("LOB Accuracy:", 
          (y_pred_lob == y_test_lob).sum() / len(y_test_lob))
    labels_lob = np.unique(np.concatenate((y_test_lob, y_pred_lob)))
    target_names_lob = le_lob.inverse_transform(labels_lob)
    print("\nLOB Report:")
    print(classification_report(
        y_test_lob, y_pred_lob, labels=labels_lob, 
        target_names=target_names_lob, zero_division=0
    ))


def save_models(models, encoders):
    """Save all models and encoders."""
    print("\nSaving models...")
    
    model_plan_name, model_plan_type, model_lob = models
    le_plan_name, le_plan_type, le_lob = encoders
    
    # Create a dictionary bundle of all models
    model_bundle = {
        'le_plan_name': le_plan_name,
        'le_plan_type': le_plan_type,
        'le_lob': le_lob,
        'model_plan_name': model_plan_name,
        'model_plan_type': model_plan_type,
        'model_lob': model_lob,
        'sentence_transformer_model_name': MODEL_NAME
    }
    
    # Save the bundle
    bundle_path = f'{MODELS_DIR}/plan_prediction_bundle.joblib'
    print(f"Saving model bundle to {bundle_path}...")
    joblib.dump(model_bundle, bundle_path)
    
    print("✓ Model bundle saved successfully!")
    print("\nSaved file:")
    print(f"  - {bundle_path}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("PLAN PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load and preprocess
    df_raw, df_clean = load_and_preprocess_data()
    
    # Fuzzy matching
    df_merged = fuzzy_match_plans(df_raw, df_clean)
    
    # Generate embeddings
    embeddings, sentence_model = generate_embeddings(df_merged)
    
    # Encode targets
    le_plan_name, le_plan_type, le_lob = encode_targets(df_merged)
    
    # Train models
    models_and_test = train_models(embeddings, df_merged)
    model_plan_name, model_plan_type, model_lob = models_and_test[:3]
    test_data = models_and_test[3:]
    
    # Evaluate
    evaluate_models(
        (model_plan_name, model_plan_type, model_lob),
        test_data,
        (le_plan_name, le_plan_type, le_lob)
    )
    
    # Save
    save_models(
        (model_plan_name, model_plan_type, model_lob),
        (le_plan_name, le_plan_type, le_lob)
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
