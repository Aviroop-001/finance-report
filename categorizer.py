# categorizer.py
import re
import os
from typing import Dict, List, Optional

CATEGORIES = [
    'Food',           # Restaurants, food delivery, cafes
    'Outing',         # Entertainment, mall food, movies, gaming
    'Shopping',       # Online/offline retail, malls (non-food)
    'Transport',      # Uber, metro, fuel, buses
    'Subscriptions',  # Netflix, Spotify, digital services
    'Home & Services', # Urban Company, home care, groceries
    'Bills',          # Utilities, broadband, mobile recharge
    'Personal',       # Payments to individuals, transfers
    'Investment',     # Groww, stocks, mutual funds
    'Finance',        # Banking, payments, financial services
    'Other'           # Fallback category
]

def categorize_with_improved_ml(transactions):
    """Categorize transactions using the improved ML model"""
    try:
        import sys
        import pandas as pd
        import joblib
        
        # Add model directory to path
        model_dir = os.path.join(os.path.dirname(__file__), 'model')
        if model_dir not in sys.path:
            sys.path.append(model_dir)
        
        # Import the model components
        from model.model_components import extract_enhanced_features
        
        model_path = os.path.join('model', 'transaction_classifier_improved.joblib')
        
        if not os.path.exists(model_path):
            print("❌ Improved ML model not found. Please run setup first to train the model.")
            return None
        
        print("🤖 Loading improved ML model...")
        pipeline = joblib.load(model_path)
        
        # Convert transactions to DataFrame
        df = pd.DataFrame(transactions)
        
        # Add category column if it doesn't exist
        if 'category' not in df.columns:
            df['category'] = 'Unknown'
        
        print("🔧 Extracting enhanced features...")
        df_enhanced = extract_enhanced_features(df)
        
        # Prepare features for prediction
        X_desc = df_enhanced['description_clean'].astype(str)
        X_merch = df_enhanced['merchant'].astype(str)
        X_amount_bins = df_enhanced['amount_bin'].astype(str)
        X_features = df_enhanced[[
            'amount', 'amount_log', 'is_sent', 'is_received', 'is_paid', 'is_transfer',
            'has_mall', 'is_food_delivery', 'is_digital_service', 'is_transport',
            'is_bills_recharge', 'is_investment', 'is_ecommerce', 'is_bank_service',
            'description_length', 'word_count', 'is_high_value', 'is_low_value',
            'is_person_name', 'is_investment_merchant', 'is_transport_merchant', 'is_grocery_merchant'
        ]].values
        
        X_combined = (X_desc, X_merch, X_amount_bins, X_features)
        
        print("🎯 Making predictions with improved model...")
        # Make predictions
        predictions = pipeline.predict(X_combined)
        probabilities = pipeline.predict_proba(X_combined)
        confidences = [max(probs) for probs in probabilities]
        
        # Update transactions with predictions
        for i, (tx, pred, conf) in enumerate(zip(transactions, predictions, confidences)):
            tx['category'] = pred
            tx['confidence'] = conf
            
            # Show some examples
            if i < 10:
                merchant = tx.get('merchant', 'Unknown')
                desc = tx.get('description', '')[:50] + "..." if len(tx.get('description', '')) > 50 else tx.get('description', '')
                print(f"   {desc} → {pred} ({conf:.1%})")
        
        print(f"✅ Categorized {len(transactions)} transactions with improved ML model (avg confidence: {sum(confidences)/len(confidences):.1%})")
        return transactions
        
    except Exception as e:
        print(f"❌ Error using improved ML model: {e}")
        print("❌ No fallback available. Please ensure the ML model is properly trained and available.")
        return None

def categorize_transactions(transactions):
    """Categorize transactions using only the improved ML model"""
    print("🤖 Categorizing transactions with improved ML model...")
    
    # Use improved ML model
    ml_result = categorize_with_improved_ml(transactions)
    if ml_result is not None:
        return ml_result
    
    # If ML model fails, return transactions without categories
    print("❌ ML model failed. Transactions will not be categorized.")
    print("📝 Please run 'python main.py setup' to train the ML model.")
    
    # Add empty categories to transactions
    for tx in transactions:
        tx['category'] = 'Uncategorized'
        tx['confidence'] = 0.0
    
    return transactions
