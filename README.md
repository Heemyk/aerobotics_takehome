# Aerobotics Missing Trees Detection API

This project is a FastAPI-based application designed to detect missing trees in orchards using geospatial data. The application fetches survey data and tree locations from an external API, processes the data, and identifies missing trees within a specified orchard boundary.

## Features

- **Fetch Surveys**: Retrieves survey data for orchards from an external API.
- **Tree Detection**: Identifies missing trees based on geospatial analysis.
- **Health Check Endpoint**: Provides a simple health check for the API.

## Endpoints

### 1. `/orchards/{orchard_id}/missing-trees`
- **Method**: `GET`
- **Description**: Fetches missing trees for the specified orchard ID.
- **Parameters**:
  - `orchard_id` (int): The ID of the orchard to analyze.
- **Response**:
  ```json
  {
    "missing_trees": [
      {"lat": <latitude>, "lng": <longitude>},
      ...
    ]
  }
  ```

### 2. `/health`
- **Method**: `GET`
- **Description**: Returns the health status of the API.
- **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Docker (optional, for containerized deployment)

### Local Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd aerobotics_takehome
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add the following:
   ```env
   API_URL=https://api.aerobotics.com
   API_KEY=<your_api_key>
   ```

5. Run the application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

6. Access the API at `http://127.0.0.1:8000`.

### Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t aerobotics-api .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 80:80 --env-file .env aerobotics-api
   ```

3. Access the API at `http://127.0.0.1`.

## Project Structure

```
aerobotics_takehome/
├── main.py               # Main FastAPI application
├── missing_point.py      # Geospatial analysis functions
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
├── .env                  # Environment variables
├── .gitignore            # Git ignore file
└── .dockerignore         # Docker ignore file
```

## Key Functions

### `get_survey_by_orchard_id(data, target_orchard_id)`
- Filters survey data to find the survey for the specified orchard ID.

### `find_missing_trees(tree_df, polygon, debug)`
- Identifies missing trees within the orchard boundary using clustering and grid-based analysis.

### `parse_polygon_string(polygon_str)`
- Parses a polygon string into a Shapely `Polygon` object.

## Dependencies

- **FastAPI**: Web framework for building APIs.
- **Requests**: For making HTTP requests to the external API.
- **Pandas**: Data manipulation and analysis.
- **Shapely**: Geospatial geometry operations.
- **PyProj**: Coordinate transformation.
- **Scikit-learn**: Clustering for tree detection.
- **Matplotlib**: Debugging and visualization.

## Testing

To test the application, you can use tools like `curl` or Postman to send requests to the endpoints. For example:

```bash
curl -X GET -H "content-type:application/json" http://127.0.0.1:8000/orchards/216269/missing-trees
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any questions or issues, please contact me at heemy.k@gmail.com.