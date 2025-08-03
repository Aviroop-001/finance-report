#!/usr/bin/env python3
"""
Enhanced Transaction Classification Model
Addresses class imbalance, adds data augmentation, and improves feature engineering
"""

import os
import re
import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy import sparse
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_components import AdvancedFeatureCombiner, extract_enhanced_features

# Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), "transaction_categorization_training_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "transaction_classifier_improved.joblib")
MIN_CLASS_SAMPLES = 3  # Minimum samples per class before consolidation
RANDOM_STATE = 42

# Import the model components from the separate module
# class AdvancedFeatureCombiner(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         # Enhanced text vectorizers
#         self.description_vectorizer = TfidfVectorizer(
#             max_features=800,
#             ngram_range=(1, 3),  # Include trigrams
#             stop_words='english',
#             min_df=2,  # Ignore terms that appear in less than 2 documents
#             max_df=0.95,  # Ignore terms that appear in more than 95% of documents
#             sublinear_tf=True  # Use sublinear tf scaling
#         )
        
#         self.merchant_vectorizer = TfidfVectorizer(
#             max_features=300,
#             ngram_range=(1, 2),
#             min_df=1,
#             max_df=0.9,
#             analyzer='word'
#         )
        
#         # Amount binning vectorizer
#         self.amount_vectorizer = TfidfVectorizer(
#             max_features=50,
#             ngram_range=(1, 1),
#             analyzer='word'
#         )
        
#         self.scaler = StandardScaler()
        
#     def fit(self, X, y=None):
#         X_desc, X_merch, X_amount_bins, X_feats = X
#         self.description_vectorizer.fit(X_desc)
#         self.merchant_vectorizer.fit(X_merch)
#         self.amount_vectorizer.fit(X_amount_bins)
#         self.scaler.fit(X_feats)
#         return self
        
#     def transform(self, X):
#         X_desc, X_merch, X_amount_bins, X_feats = X
        
#         # Transform each feature type
#         desc_features = self.description_vectorizer.transform(X_desc)
#         merch_features = self.merchant_vectorizer.transform(X_merch)
#         amount_features = self.amount_vectorizer.transform(X_amount_bins)
#         numeric_features = self.scaler.transform(X_feats)
        
#         # Combine all features
#         combined = sparse.hstack([
#             desc_features, 
#             merch_features, 
#             amount_features,
#             numeric_features
#         ])
#         return combined

def smart_class_consolidation(df, min_samples=MIN_CLASS_SAMPLES):
    """Intelligently consolidate similar classes and handle rare classes"""
    
    # Count samples per class
    class_counts = df['category'].value_counts()
    print(f"📊 Original class distribution:")
    for cat, count in class_counts.items():
        print(f"  {cat:20}: {count:3d} samples")
    
    # Define consolidation rules for similar categories
    consolidation_rules = {
        'HomeMaintenance': 'Home & Services',
        'Pharmacy': 'Home & Services',
        'PeerPayment': 'Transfer',  # Only 1 sample, merge with Transfer
    }
    
    # Apply consolidation rules
    df['category_original'] = df['category'].copy()
    for old_cat, new_cat in consolidation_rules.items():
        mask = df['category'] == old_cat
        df.loc[mask, 'category'] = new_cat
        if mask.sum() > 0:
            print(f"📝 Consolidated '{old_cat}' ({mask.sum()} samples) into '{new_cat}'")
    
    # Handle remaining rare classes
    updated_counts = df['category'].value_counts()
    rare_classes = updated_counts[updated_counts < min_samples].index
    
    if len(rare_classes) > 0:
        print(f"\n⚠️  Merging rare classes (< {min_samples} samples) into 'Other':")
        for rare_class in rare_classes:
            count = updated_counts[rare_class]
            print(f"  {rare_class}: {count} samples")
            df.loc[df['category'] == rare_class, 'category'] = 'Other'
    
    # Final distribution
    final_counts = df['category'].value_counts()
    print(f"\n📊 Final class distribution:")
    for cat, count in final_counts.items():
        print(f"  {cat:20}: {count:3d} samples")
    
    return df

def create_ensemble_model():
    """Create an ensemble model combining multiple algorithms"""
    
    # Individual models with different strengths
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight='balanced'  # Handle class imbalance
    )
    
    lgb_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=10,
        num_leaves=31,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        verbosity=-1
    )
    
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        C=1.0
    )
    
    # Ensemble with voting
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('lgb', lgb_model),
            ('lr', lr_model)
        ],
        voting='soft'  # Use probability averaging
    )
    
    return ensemble

def train_improved_model():
    """Train the improved model with all enhancements"""
    
    # Load and prepare data
    df = pd.read_csv(DATA_PATH)
    if not {'description', 'category'}.issubset(df.columns):
        raise ValueError("CSV must have 'description' and 'category' columns")
    
    print(f"📊 Loaded {len(df)} transactions")
    
    # Feature engineering
    from model_components import extract_enhanced_features
    df = extract_enhanced_features(df)
    df = smart_class_consolidation(df, min_samples=MIN_CLASS_SAMPLES)
    
    # Prepare features
    X_desc = df['description_clean'].astype(str)
    X_merch = df['merchant'].astype(str)
    X_amount_bins = df['amount_bin'].astype(str)
    X_features = df[[
        'amount', 'amount_log', 'is_sent', 'is_received', 'is_paid', 'is_transfer',
        'has_mall', 'is_food_delivery', 'is_digital_service', 'is_transport',
        'is_bills_recharge', 'is_investment', 'is_ecommerce', 'is_bank_service',
        'description_length', 'word_count', 'is_high_value', 'is_low_value',
        'is_person_name', 'is_investment_merchant', 'is_transport_merchant', 'is_grocery_merchant'
    ]].values
    
    y = df['category']
    
    # Create the pipeline with SMOTE for oversampling
    pipeline = ImbPipeline([
        ('features', AdvancedFeatureCombiner()),
        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=2)),  # Oversample minority classes
        ('classifier', create_ensemble_model())
    ])
    
    # Custom cross-validation with stratified folds
    print("\n🔄 Performing cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Prepare data for CV
    X_combined = (X_desc, X_merch, X_amount_bins, X_features)
    
    # Custom CV loop to handle tuple input format
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_desc, y)):
        print(f"  Fold {fold + 1}/5...")
        
        # Split data for this fold
        X_desc_fold = X_desc.iloc[train_idx]
        X_merch_fold = X_merch.iloc[train_idx]
        X_amount_fold = X_amount_bins.iloc[train_idx]
        X_feat_fold = X_features[train_idx]
        y_fold = y.iloc[train_idx]
        
        X_desc_val = X_desc.iloc[val_idx]
        X_merch_val = X_merch.iloc[val_idx]
        X_amount_val = X_amount_bins.iloc[val_idx]
        X_feat_val = X_features[val_idx]
        y_val = y.iloc[val_idx]
        
        # Create pipeline for this fold
        fold_pipeline = ImbPipeline([
            ('features', AdvancedFeatureCombiner()),
            ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=2)),
            ('classifier', create_ensemble_model())
        ])
        
        # Train and validate
        X_train_fold = (X_desc_fold, X_merch_fold, X_amount_fold, X_feat_fold)
        X_val_fold = (X_desc_val, X_merch_val, X_amount_val, X_feat_val)
        
        fold_pipeline.fit(X_train_fold, y_fold)
        y_pred_fold = fold_pipeline.predict(X_val_fold)
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val, y_pred_fold, average='weighted', zero_division=0)
        cv_scores.append(f1)
    
    cv_scores = np.array(cv_scores)
    print(f"📈 5-fold CV F1-weighted: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Train/test split for final evaluation
    # Generate indices first to ensure consistent splits
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        range(len(y)), test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Apply the same indices to all feature types
    X_desc_train, X_desc_test = X_desc.iloc[train_idx], X_desc.iloc[test_idx]
    X_merch_train, X_merch_test = X_merch.iloc[train_idx], X_merch.iloc[test_idx]
    X_amount_train, X_amount_test = X_amount_bins.iloc[train_idx], X_amount_bins.iloc[test_idx]
    X_feat_train, X_feat_test = X_features[train_idx], X_features[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    X_train_combined = (X_desc_train, X_merch_train, X_amount_train, X_feat_train)
    X_test_combined = (X_desc_test, X_merch_test, X_amount_test, X_feat_test)
    
    # Train the model
    print("\n🚀 Training improved model...")
    pipeline.fit(X_train_combined, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test_combined)
    y_pred_proba = pipeline.predict_proba(X_test_combined)
    
    # Evaluation
    print("\n🧪 Final Test Results:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()
    print(f"\n📊 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=pipeline.classes_, 
                yticklabels=pipeline.classes_)
    plt.title('Confusion Matrix - Improved Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance analysis (for Random Forest component)
    if hasattr(pipeline.named_steps['classifier'].named_estimators_['rf'], 'feature_importances_'):
        print("\n🎯 Feature Importance Analysis:")
        
        # Get feature names
        feature_names = (
            [f'desc_{i}' for i in range(800)] +
            [f'merch_{i}' for i in range(300)] +
            [f'amount_{i}' for i in range(50)] +
            ['amount', 'amount_log', 'is_sent', 'is_received', 'is_paid', 'is_transfer',
             'has_mall', 'is_food_delivery', 'is_digital_service', 'is_transport',
             'is_bills_recharge', 'is_investment', 'is_ecommerce', 'is_bank_service',
             'description_length', 'word_count', 'is_high_value', 'is_low_value']
        )
        
        importances = pipeline.named_steps['classifier'].named_estimators_['rf'].feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
        
        print("Top 20 Most Important Features:")
        for feature, importance in top_features:
            print(f"  {feature:30}: {importance:.4f}")
    
    # Save the model
    os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Improved model saved to: {MODEL_PATH}")
    
    return pipeline, accuracy

def test_improved_model(transaction_text):
    """Test the improved model on a single transaction"""
    if not os.path.exists(MODEL_PATH):
        print("❌ Improved model not found. Run training first.")
        return
    
    pipeline = joblib.load(MODEL_PATH)
    
    # Create temporary dataframe for feature extraction
    temp_df = pd.DataFrame({'description': [transaction_text], 'category': ['Unknown']})
    temp_df = extract_enhanced_features(temp_df)
    
    # Prepare features
    X_desc = temp_df['description_clean']
    X_merch = temp_df['merchant']
    X_amount_bins = temp_df['amount_bin']
    X_features = temp_df[[
        'amount', 'amount_log', 'is_sent', 'is_received', 'is_paid', 'is_transfer',
        'has_mall', 'is_food_delivery', 'is_digital_service', 'is_transport',
        'is_bills_recharge', 'is_investment', 'is_ecommerce', 'is_bank_service',
        'description_length', 'word_count', 'is_high_value', 'is_low_value',
        'is_person_name', 'is_investment_merchant', 'is_transport_merchant', 'is_grocery_merchant'
    ]].values
    
    X_combined = (X_desc, X_merch, X_amount_bins, X_features)
    
    # Predict
    prediction = pipeline.predict(X_combined)[0]
    probabilities = pipeline.predict_proba(X_combined)[0]
    confidence = max(probabilities)
    
    print(f"📝 Transaction: {transaction_text}")
    print(f"💡 Extracted merchant: {temp_df['merchant'].iloc[0]}")
    print(f"💰 Amount: ₹{temp_df['amount'].iloc[0]}")
    print(f"🎯 Predicted Category: {prediction}")
    print(f"🔢 Confidence: {confidence:.2%}")
    
    # Show top 3 predictions
    classes = pipeline.classes_
    top_indices = probabilities.argsort()[-3:][::-1]
    print(f"\n📊 Top 3 predictions:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {classes[idx]}: {probabilities[idx]:.2%}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Transaction Classification Model")
    parser.add_argument('--train', action='store_true', help='Train the improved model')
    parser.add_argument('--test', type=str, help='Test single transaction')
    
    args = parser.parse_args()
    
    if args.train:
        pipeline, accuracy = train_improved_model()
        print(f"\n🎉 Training completed! Final accuracy: {accuracy:.1%}")
    elif args.test:
        test_improved_model(args.test)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 