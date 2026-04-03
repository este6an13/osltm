from fastapi import APIRouter
import os
import sqlite3

router = APIRouter()

DB_PATH = "d:/dequi/repositories/osltm/osltm.db" # Check if using osltm.db or osltm_v2.db

@router.get("/")
def get_system_status():
    status = {
        "db_size_mb": 0,
        "sampled_dates": 0,
        "sampled_stations": 0,
        "runs": []
    }
    
    if os.path.exists(DB_PATH):
        status["db_size_mb"] = round(os.path.getsize(DB_PATH) / (1024 * 1024), 2)
        
    dates_file = "d:/dequi/repositories/osltm/src/workflow/data/sampled_dates.csv"
    if os.path.exists(dates_file):
        with open(dates_file, 'r') as f:
            status["sampled_dates"] = sum(1 for line in f) - 1 # exclude header
            
    stations_file = "d:/dequi/repositories/osltm/src/workflow/data/sampled_stations.csv"
    if os.path.exists(stations_file):
        with open(stations_file, 'r', encoding='utf-8') as f:
            status["sampled_stations"] = sum(1 for line in f) - 1
            
    # Add runs history here from runner service later
    
    return status
