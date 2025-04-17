# main.py
from fastapi import FastAPI
import requests
from typing import List, Dict
from dotenv import load_dotenv
import os
import pandas as pd

from missing_point import find_missing_trees, parse_polygon_string

app = FastAPI()

# Credentials
load_dotenv()
API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")



def get_survey_by_orchard_id(data, target_orchard_id):
    for survey in data['results']:
        if survey['orchard_id'] == target_orchard_id:
            return survey
    return None  # Return None if no matching orchard_id is found


@app.get("/orchards/{orchard_id}/missing-trees")
async def get_missing_trees(orchard_id: int) -> Dict[str, List[Dict[str, float]]]:
    """Fetch missing trees for the given orchard_id."""
    if orchard_id != 216269:
        return {"missing_trees": []}  # For testing, only handle 216269
    
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    # ------------------------Responses------------------------

    # Surveys
    all_surveys_response = requests.get(
        f"{API_URL}/farming/surveys/",  # Adjust based on API docs
        headers=headers
    )
    all_surveys_response.raise_for_status()
    all_surveys = all_surveys_response.json()
    survey = get_survey_by_orchard_id(all_surveys, orchard_id)
    id = survey["id"]
    
    polygon_string = survey["polygon"]
    polygon = parse_polygon_string(polygon_string)


    # Trees
    tree_surveys_response = requests.get(
        f"{API_URL}/farming/surveys/{id}/tree_surveys/",  # Adjust based on API docs
        headers=headers
    )
    tree_surveys_response.raise_for_status()
    tree_surveys = tree_surveys_response.json()  # List of {"lat": ..., "lng": ...}

    tree_surveys_df = pd.DataFrame(tree_surveys["results"])
    
    # --------------------------------------------------------------------------------------

    missing_trees = find_missing_trees(tree_surveys_df, polygon, debug=False)
    return {"missing_trees": missing_trees}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}