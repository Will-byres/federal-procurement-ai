import spacy
import pandas as pd

# Load small English model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> dict:
    """Extract key named entities from contract text."""
    if not isinstance(text, str) or not text.strip():
        return {}
    
    doc = nlp(text)
    entities = {
        "Organizations": list({ent.text for ent in doc.ents if ent.label_ == "ORG"}),
        "Locations": list({ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]}),
        "Products_Facilities": list({ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "FAC"]}),
        "Dates_Deadlines": list({ent.text for ent in doc.ents if ent.label_ == "DATE"})
    }
    return entities

if __name__ == "__main__":
    print("Loading cleaned dataset...")
    df = pd.read_csv("data/cleaned_contracts.csv")
    
    # Test on a sample of 5 contracts
    print("\n--- Sample NER Extraction ---")
    sample_descriptions = df["Cleaned_Description"].dropna().head(5)
    
    for idx, desc in enumerate(sample_descriptions, start=1):
        print(f"\n[Contract #{idx}]: {desc[:100]}...")
        extracted = extract_entities(desc)
        for label, values in extracted.items():
            if values:
                print(f"  • {label}: {', '.join(values)}")