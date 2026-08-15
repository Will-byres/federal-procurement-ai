import pandas as pd
import re

def clean_procurement_text(text):
    """
    Cleans raw contract descriptions for NLP modeling.
    Removes boilerplate government tags, codes, and excessive whitespace.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # 1. Remove common government contract boilerplate prefixes (e.g., IGF::OT::IGF, TAS::..., etc.)
    text = re.sub(r'IGF::[A-Z0-9:]+::IGF', '', text)
    text = re.sub(r'TAS::[A-Z0-9:]+', '', text)
    text = re.sub(r'N00\d+-\d+-\w-\d+', '', text) # Common DoD contract numbers inside text
    
    # 2. Remove special characters and noise symbols, keep alphanumeric and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,$-]', ' ', text)
    
    # 3. Collapse multiple spaces / newlines into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def run_pipeline():
    input_file = "data/raw_contracts.csv"
    output_file = "data/cleaned_contracts.csv"
    
    print(f"Loading raw data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Total raw records loaded: {len(df)}")
    
    # 1. Drop records where Description is missing or null
    df = df.dropna(subset=['Description'])
    
    # 2. Clean the Description column
    print("Cleaning contract descriptions for NLP...")
    df['Cleaned_Description'] = df['Description'].apply(clean_procurement_text)
    
    # 3. Filter out very short/useless descriptions (less than 15 characters)
    df = df[df['Cleaned_Description'].str.len() > 15]
    
    # 4. Standardize numeric and date columns
    df['Award Amount'] = pd.to_numeric(df['Award Amount'], errors='coerce')
    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
    df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
    
    # 5. Save cleaned dataset
    df.to_csv(output_file, index=False)
    print(f"\nTransformation complete! {len(df)} cleaned records saved to {output_file}")
    
    # Display preview
    print("\nSample Cleaned Data:")
    print(df[['Recipient Name', 'Award Amount', 'Cleaned_Description']].head(3))

if __name__ == "__main__":
    run_pipeline()