"""
Model components for the improved transaction classifier
This module contains the classes and functions needed to load the saved model
"""

import re
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy import sparse

class AdvancedFeatureCombiner(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        # Enhanced text vectorizers
        self.description_vectorizer = TfidfVectorizer(
            max_features=800,
            ngram_range=(1, 3),  # Include trigrams
            stop_words='english',
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True  # Use sublinear tf scaling
        )
        
        self.merchant_vectorizer = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            analyzer='word'
        )
        
        # Amount binning vectorizer
        self.amount_vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 1),
            analyzer='word'
        )
        
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        X_desc, X_merch, X_amount_bins, X_feats = X
        self.description_vectorizer.fit(X_desc)
        self.merchant_vectorizer.fit(X_merch)
        self.amount_vectorizer.fit(X_amount_bins)
        self.scaler.fit(X_feats)
        return self
        
    def transform(self, X):
        X_desc, X_merch, X_amount_bins, X_feats = X
        
        # Transform each feature type
        desc_features = self.description_vectorizer.transform(X_desc)
        merch_features = self.merchant_vectorizer.transform(X_merch)
        amount_features = self.amount_vectorizer.transform(X_amount_bins)
        numeric_features = self.scaler.transform(X_feats)
        
        # Combine all features
        combined = sparse.hstack([
            desc_features, 
            merch_features, 
            amount_features,
            numeric_features
        ])
        return combined

def extract_enhanced_features(df):
    """Extract enhanced features from transaction data"""
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Clean description text
    df['description_clean'] = df['description'].fillna('').astype(str).apply(clean_text)
    
    # Extract merchant names
    df['merchant'] = df['description'].fillna('').astype(str).apply(extract_merchant_name)
    
    # Basic numeric features
    df['amount'] = df['description'].fillna('').apply(extract_amount).astype(float)
    df['amount_log'] = np.log1p(df['amount'])
    
    # Transaction type flags
    df['is_sent'] = df['description'].str.contains(r'sent|transferred', case=False, regex=True).astype(int)
    df['is_received'] = df['description'].str.contains(r'received|credited', case=False, regex=True).astype(int)
    df['is_paid'] = df['description'].str.contains(r'paid|payment', case=False, regex=True).astype(int)
    df['is_transfer'] = df['description'].str.contains(r'transfer|sent|received', case=False, regex=True).astype(int)
    
    # Context flags
    df['has_mall'] = df['description'].str.contains(r'mall|forum|nexus|phoenix|orion', case=False, regex=True).astype(int)
    df['is_food_delivery'] = df['description'].str.contains(r'swiggy|zomato|food|delivery', case=False, regex=True).astype(int)
    df['is_digital_service'] = df['description'].str.contains(r'netflix|spotify|amazon|prime|hotstar', case=False, regex=True).astype(int)
    
    # Transport patterns
    transport_patterns = [
        # Travel booking platforms
        r'Goibibo|goibibo|ibibo|makemytrip|cleartrip|easemytrip|yatra',
        # Airlines and flights
        r'indigo|air.*india|spicejet|vistara|flight|airline',
        # Ground transport
        r'uber|ola|rapido|metro|bus|train|irctc|railway|transport|travel',
        # Car related
        r'petrol|diesel|fuel|parking|toll',
        # Generic travel terms
        r'booking|ticket'
    ]
    df['is_transport'] = df['description'].str.contains('|'.join(transport_patterns), case=False, regex=True).astype(int) * 2
    
    # Bills and recharge patterns
    df['is_bills_recharge'] = df['description'].str.contains(r'recharge|bill|airtel|jio|electricity|broadband', case=False, regex=True).astype(int)
    
    # Investment patterns
    df['is_investment'] = df['description'].str.contains(r'groww|zerodha|upstox|mutual|fund|stock|investment', case=False, regex=True).astype(int)
    
    # E-commerce patterns
    df['is_ecommerce'] = df['description'].str.contains(r'amazon|flipkart|myntra|ajio|meesho', case=False, regex=True).astype(int)
    
    # Bank service patterns
    df['is_bank_service'] = df['description'].str.contains(r'bank|charge|fee|interest|emi|loan', case=False, regex=True).astype(int)
    
    # Text-based features
    df['description_length'] = df['description'].str.len()
    df['word_count'] = df['description'].str.count(r'\s+') + 1
    
    # Amount-based flags
    amount_mean = df['amount'].mean()
    amount_std = df['amount'].std()
    df['is_high_value'] = (df['amount'] > (amount_mean + amount_std)).astype(int)
    df['is_low_value'] = (df['amount'] < (amount_mean - amount_std)).astype(int)
    
    # Person name detection
    df['is_person_name'] = df['description'].apply(lambda x: contains_person_name(x)).astype(int)
    
    # Merchant type detection
    investment_patterns = [
        r'groww|zerodha|upstox|mutual|fund|stock|investment|nextbillion|iccl|demat'
    ]
    df['is_investment_merchant'] = df['description'].str.contains('|'.join(investment_patterns), case=False, regex=True).astype(int)
    
    transport_merchant_patterns = [
        # Travel booking platforms
        r'Goibibo|goibibo|ibibo|makemytrip|cleartrip|easemytrip|yatra',
        # Airlines
        r'indigo|air.*india|spicejet|vistara',
        # Ground transport
        r'uber|ola|rapido|metro|railway|irctc|transport',
        # Car services
        r'petrol|fuel|parking|toll',
        # Generic travel
        r'travel|booking'
    ]
    df['is_transport_merchant'] = df['description'].str.contains('|'.join(transport_merchant_patterns), case=False, regex=True).astype(int) * 2
    
    grocery_patterns = [
        r'blinkit|bigbasket|grofers|dunzo|zepto|dmart|reliance|fresh|nature|basket|grocery|supermarket'
    ]
    df['is_grocery_merchant'] = df['description'].str.contains('|'.join(grocery_patterns), case=False, regex=True).astype(int)
    
    # Amount binning
    try:
        # Try to create 10 bins
        df['amount_bin'] = pd.qcut(df['amount'], q=10, labels=[f'bin_{i}' for i in range(10)], duplicates='drop')
    except ValueError:
        # If that fails due to too many duplicates, try 5 bins
        try:
            df['amount_bin'] = pd.qcut(df['amount'], q=5, labels=[f'bin_{i}' for i in range(5)], duplicates='drop')
        except ValueError:
            # If that still fails, use fixed bins
            df['amount_bin'] = pd.cut(
                df['amount'],
                bins=[-float('inf'), 100, 500, 1000, 5000, float('inf')],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
    
    df['amount_bin'] = df['amount_bin'].astype(str)
    
    return df

def clean_text(text):
    """Clean and normalize text"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_merchant_name(text):
    """Extract merchant name from transaction description"""
    # Look for "to" pattern
    to_match = re.search(r'to\s+([^using]+?)(?:\s+using|$)', str(text), re.IGNORECASE)
    if to_match:
        return to_match.group(1).strip()
    
    # Look for common merchant patterns
    merchant_match = re.search(r'(?:at|from|via|through)\s+([^using]+?)(?:\s+using|$)', str(text), re.IGNORECASE)
    if merchant_match:
        return merchant_match.group(1).strip()
    
    return ''

def extract_amount(text):
    """Extract amount from transaction description"""
    amount_match = re.search(r'₹([\d,]+\.?\d*)', str(text))
    if amount_match:
        amount_str = amount_match.group(1).replace(',', '')
        return float(amount_str)
    return 0.0

def contains_person_name(text):
    """Check if description likely contains a person's name"""
    # Common Indian name patterns
    name_patterns = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
        r'\b[A-Z][A-Z\s]+\b',  # FULL NAME
        r'\b[A-Z][a-z]+ [A-Z]\b',  # First I.
        r'\b[A-Z] [A-Z][a-z]+\b',  # I. Last
    ]
    
    # Check for UPI-style IDs
    upi_pattern = r'@[a-zA-Z0-9\.]+'
    
    for pattern in name_patterns:
        if re.search(pattern, str(text)):
            return True
            
    # Check for UPI ID
    if re.search(upi_pattern, str(text)):
        return True
        
    return False 