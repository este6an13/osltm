from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
import pandas as pd
from typing import List, Dict

router = APIRouter()

RESULTS_DIR = "d:/dequi/repositories/osltm/src/workflow/results"

@router.get("/")
def list_result_directories():
    if not os.path.exists(RESULTS_DIR):
        return []
    dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    return sorted(dirs)

@router.get("/{directory}")
def list_directory_files(directory: str):
    dir_path = os.path.join(RESULTS_DIR, directory)
    if not os.path.exists(dir_path) or not os.path.abspath(dir_path).startswith(os.path.abspath(RESULTS_DIR)):
        raise HTTPException(status_code=404, detail="Directory not found")
        
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    return sorted(files)

@router.get("/{directory}/{filename}/view")
def view_file(directory: str, filename: str):
    file_path = os.path.join(RESULTS_DIR, directory, filename)
    if not os.path.exists(file_path) or not os.path.abspath(file_path).startswith(os.path.abspath(RESULTS_DIR)):
        raise HTTPException(status_code=404, detail="File not found")
        
    if filename.endswith('.csv'):
        # Just return the raw file, let frontend parse it or we can return json
        df = pd.read_csv(file_path)
        # Handle Inf / NaN
        df = df.replace([float('inf'), float('-inf')], None).where(pd.notnull(df), None)
        return {
            "columns": df.columns.tolist(),
            "data": df.to_dict('records')
        }
    else:
        return FileResponse(file_path)

@router.get("/{directory}/{filename}/download")
def download_file(directory: str, filename: str):
    file_path = os.path.join(RESULTS_DIR, directory, filename)
    if not os.path.exists(file_path) or not os.path.abspath(file_path).startswith(os.path.abspath(RESULTS_DIR)):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, filename=filename)
