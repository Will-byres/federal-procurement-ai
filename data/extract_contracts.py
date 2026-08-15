import requests
import pandas as pd
import time

def fetch_usaspending_contracts(max_pages=50):
    """
    Fetches contract data from the USAspending API.
    max_pages: The number of pages to pull (100 records per page).
    """
    
    url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
    
    # The fields we want the API to return
    fields_to_request = [
        "Award ID", 
        "Recipient Name", 
        "Award Amount", 
        "Awarding Agency", 
        "Description",
        "Start Date",
        "End Date"
    ]

    all_results = []
    
    print(f"Starting data extraction. Fetching up to {max_pages} pages...")

    for page in range(1, max_pages + 1):
        print(f"Fetching page {page}...")
        
        payload = {
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"] # Filters for distinct contract types
            },
            "fields": fields_to_request,
            "limit": 100, # Max allowed limit per request
            "page": page,
            "sort": "Award Amount",
            "order": "desc"
        }
        
        headers = {
            "Content-Type": "application/json"
        }

        try:
            # POST request is leveraged here because advanced filtering is required
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status() # Raise an error for bad status codes
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                print("No more results found. Ending pagination.")
                break
                
            all_results.extend(results)
            
            # Small delay to be polite to the government servers
            time.sleep(1) 
            
        except requests.exceptions.RequestException as e:
            print(f"An error occurred on page {page}: {e}")
            break

    return all_results

if __name__ == "__main__":
    # 1. Fetch the data
    # Change max_pages to a higher number to get more data
    raw_data = fetch_usaspending_contracts(max_pages=50) 
    
    if raw_data:
        # 2. Convert to a Pandas DataFrame for easy manipulation
        df = pd.DataFrame(raw_data)
        
        # 3. Save to the data folder
        output_path = "data/raw_contracts.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSuccess! Successfully saved {len(df)} records to {output_path}")
    else:
        print("\nNo data was extracted.")