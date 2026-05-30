"""
Step 7: Extract TransMilenio routes frequencies from Tullave.

This step fetches all active routes of a specified service type (e.g., Troncal)
from the Tullave frequencies microservice API, then queries the details page
for each route sequentially to parse and structure operational frequencies,
destinations, and operating times, saving them to a CSV file.
"""

import os
import json
import csv
import urllib.request
import urllib.parse
import time
import sys
import re
from pathlib import Path
from typing import Any

BASE_URL = "https://frecuencias.tullaveplus.gov.co"

def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            enc = sys.stdout.encoding or 'utf-8'
            print(msg.encode(enc, errors='replace').decode(enc))
        except Exception:
            print(msg.encode('ascii', errors='replace').decode('ascii'))

def make_request(url, timeout=120, retries=2):
    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode('utf-8')
        except Exception as e:
            safe_print(f"  Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                time.sleep(attempt * 3)
    return None

def fetch_route_data(route_id, route_name, tipo):
    query_params = urllib.parse.urlencode({
        "tab": "frecuencias",
        "tipoServicio": tipo,
        "ruta": route_id
    })
    url = f"{BASE_URL}/?{query_params}"
    
    start_t = time.time()
    html = make_request(url)
    elapsed = time.time() - start_t
    
    if not html:
        safe_print(f"  Route {route_name} (ID: {route_id}) failed completely in {elapsed:.2f}s")
        # Return a placeholder row
        placeholder = [
            route_id,
            tipo,
            route_name,
            "---",
            "---",
            "---",
            "No hay Información",
            "---",
            "SÍ"
        ]
        return [placeholder]
        
    table_match = re.search(r'<table id="mainTable"[^>]*>.*?<tbody>(.*?)</tbody>.*?</table>', html, re.DOTALL)
    if not table_match:
        safe_print(f"  Route {route_name} (ID: {route_id}) has no mainTable in {elapsed:.2f}s")
        placeholder = [
            route_id,
            tipo,
            route_name,
            "---",
            "---",
            "---",
            "No hay Información",
            "---",
            "SÍ"
        ]
        return [placeholder]
        
    tbody = table_match.group(1)
    rows = re.findall(r'<tr>(.*?)</tr>', tbody, re.DOTALL)
    
    extracted_rows = []
    for r in rows:
        cols = re.findall(r'<td[^>]*>(.*?)</td>', r, re.DOTALL)
        cols_clean = [re.sub(r'<[^>]*>', '', c).strip() for c in cols]
        if len(cols_clean) == 9:
            cols_clean[8] = 'SÍ' if 'S' in cols_clean[8] else 'NO'
            extracted_rows.append(cols_clean)
            
    safe_print(f"  Route {route_name} (ID: {route_id}) succeeded: extracted {len(extracted_rows)} directions in {elapsed:.2f}s")
    
    if not extracted_rows:
        placeholder = [
            route_id,
            tipo,
            route_name,
            "---",
            "---",
            "---",
            "No hay Información",
            "---",
            "SÍ"
        ]
        return [placeholder]
        
    return extracted_rows

def run(params: dict[str, Any]) -> None:
    """
    Execute step 7: Extract route frequencies and hours sequentially.
    """
    step_params = params.get("step7", {})
    output_dir_str = step_params.get("output_dir", "data/routes")
    output_filename = step_params.get("output_filename", "transmilenio_frequencies.csv")
    tipo = step_params.get("tipo", "Troncal")
    sleep_delay = step_params.get("sleep_delay", 1.5)
    
    workspace_root = Path(__file__).parent.parent.parent.parent.resolve()
    if Path(output_dir_str).is_absolute():
        output_dir = Path(output_dir_str)
    else:
        output_dir = workspace_root / output_dir_str
        
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename
    
    safe_print("--- Step 7: Tullave Route Frequencies Extractor (Polite Sequential) ---")
    safe_print(f"📁 Output Directory: {output_dir.resolve()}")
    safe_print(f"📄 Output File: {output_file.resolve()}")
    safe_print(f"🏷️ Service Type: {tipo}")
    safe_print(f"⏱️ Sleep Delay: {sleep_delay}s")
    
    # 1. Fetch routes list
    routes_url = f"{BASE_URL}/api/rutas?tipo={urllib.parse.quote(tipo)}&tab=frecuencias"
    safe_print("Fetching route list from API...")
    routes_json = make_request(routes_url, timeout=30, retries=3)
    if not routes_json:
        raise RuntimeError("Failed to retrieve routes list from Tullave API.")
        
    try:
        routes = json.loads(routes_json)
    except Exception as ex:
        raise RuntimeError(f"Failed to parse routes JSON: {ex}")
        
    total_routes = len(routes)
    safe_print(f"Successfully retrieved {total_routes} routes.")
    
    # 2. Check existing CSV for non-empty progress
    existing_ids = set()
    existing_rows = []
    headers = [
        "line_id", "type", "name", "destination", 
        "frequency_peak", "frequency_offpeak", 
        "operation_days", "schedule", "accessible"
    ]
    
    if output_file.exists():
        try:
            with open(output_file, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                for r in reader:
                    if r:
                        # If a route has actual frequencies (not '---'), we mark its master route ID as done
                        if r[4] != '---' and r[4] != '':
                            # We need to map back to the master route ID
                            # For safety, let's find the matching route from routes list by name
                            matching_route = next((x for x in routes if x.get("nombre") == r[2]), None)
                            if matching_route:
                                existing_ids.add(matching_route.get("id"))
                        existing_rows.append(r)
            safe_print(f"Loaded {len(existing_rows)} rows. Found {len(existing_ids)} routes already successfully scraped with actual frequencies.")
        except Exception as e:
            safe_print(f"Warning: Error reading existing CSV: {e}. Starting fresh.")
            
    # Filter out already successfully scraped routes
    missing_routes = [r for r in routes if r.get("id") not in existing_ids]
    safe_print(f"Number of routes that need scraping/updating: {len(missing_routes)}")
    
    if not missing_routes:
        safe_print("All routes are already successfully scraped with frequencies!")
        return
        
    # Remove any existing rows for the missing routes to prevent duplicates
    missing_names = set(r.get("nombre") for r in missing_routes)
    current_results = [r for r in existing_rows if not (r[1].upper() == tipo.upper() and r[2] in missing_names)]
    
    # 3. Scrape sequentially and write row-by-row
    for idx, r in enumerate(missing_routes, start=1):
        r_id = r.get("id")
        r_name = r.get("nombre")
        
        safe_print(f"[{idx}/{len(missing_routes)}] Route: {r_name} (ID: {route_id if 'route_id' in locals() else r_id})...")
        
        rows = fetch_route_data(r_id, r_name, tipo)
        current_results.extend(rows)
        
        # Sort by line_id (int if possible)
        temp_rows = list(current_results)
        try:
            temp_rows.sort(key=lambda x: int(x[0]))
        except Exception:
            try:
                temp_rows.sort(key=lambda x: x[0])
            except Exception:
                pass
                
        # Save progress back to CSV immediately
        try:
            with open(output_file, mode='w', encoding='utf-8', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(headers)
                writer.writerows(temp_rows)
        except Exception as e:
            safe_print(f"  Warning: Failed to write temp CSV: {e}")
            
        time.sleep(sleep_delay)
        
    safe_print(f"✅ Frequencies extraction completed! CSV saved at: {output_file}")
    safe_print(f"📊 Total unique rows in CSV: {len(current_results)}")

if __name__ == "__main__":
    params_path = Path(__file__).parent.parent / "params.json"
    if params_path.exists():
        with open(params_path) as f:
            params_dict = json.load(f)
        run(params_dict)
