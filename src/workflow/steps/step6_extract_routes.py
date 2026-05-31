"""
Step 6: Extract TransMilenio routes and stations.

This step fetches all active routes of a specified service type and their
sequential stops from the official TransMilenio microservice API, saving
the compiled details to a CSV file.
"""

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://ms-transmiapp-rm2xahnybq-uk.a.run.app"


def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            enc = sys.stdout.encoding or "utf-8"
            print(msg.encode(enc, errors="replace").decode(enc))
        except Exception:
            print(msg.encode("ascii", errors="replace").decode("ascii"))


def make_request(url, data=None, method="GET"):
    req = urllib.request.Request(url, method=method)
    req.add_header(
        "User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    req.add_header("Accept", "application/json")

    req_data = None
    if data is not None:
        req.add_header("Content-Type", "application/json")
        req_data = json.dumps(data).encode("utf-8")

    try:
        with urllib.request.urlopen(req, data=req_data) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        safe_print(f"Error requesting {url}: {e}")
        return None


def run(params: dict[str, Any]) -> None:
    """
    Execute step 6: Extract routes and stations.
    """
    import datetime
    step_params = params.get("step6", {})
    output_dir_str = step_params.get("output_dir", "data/routes")
    output_filename = step_params.get(
        "output_filename", "transmilenio_routes_stations.csv"
    )
    
    # Append timestamp before extension to prevent overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if "." in output_filename:
        parts = output_filename.rsplit(".", 1)
        output_filename = f"{parts[0]}_{timestamp}.{parts[1]}"
    else:
        output_filename = f"{output_filename}_{timestamp}"

    tipo = step_params.get("tipo", "TransMilenio")
    sleep_delay = max(0.5, float(step_params.get("sleep_delay", 0.5)))

    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename

    safe_print("--- Step 6: TransMilenio Route & Station Extractor ---")
    safe_print(f"📁 Output Directory: {output_dir.resolve()}")
    safe_print(f"📄 Output File: {output_file.resolve()}")
    safe_print(f"🏷️ Service Type: {tipo}")
    safe_print(f"⏱️ Sleep Delay: {sleep_delay}s")

    # 1. Fetch routes
    search_params = urllib.parse.urlencode(
        {"page": 0, "size": 1000, "sort": "idCodigo,asc"}
    )
    search_url = f"{BASE_URL}/api/v1/rutas/buscar?{search_params}"
    search_body = {
        "tipo": tipo,
        "troncalId": None,
        "estacionId": None,
        "zona": None,
        "q": None,
    }

    safe_print("Fetching route list from API...")
    search_response = make_request(search_url, data=search_body, method="POST")
    if not search_response or "content" not in search_response:
        raise RuntimeError("Failed to retrieve routes list from TransMilenio API.")

    routes = search_response["content"]
    total_routes = len(routes)
    safe_print(f"Successfully retrieved {total_routes} routes.")

    # 2. Write CSV
    headers = [
        "route_id",
        "route_code",
        "route_name",
        "route_color",
        "troncal_id",
        "troncal_name",
        "troncal_zone",
        "schedule_mon_sat",
        "schedule_sun_hol",
        "station_sequence",
        "station_id",
        "station_code",
        "station_name",
        "station_address",
    ]

    try:
        with open(output_file, mode="w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)

            for idx, route in enumerate(routes, start=1):
                route_id = route.get("id")
                route_code = route.get("codigo")
                route_name = route.get("nombre")
                route_color = route.get("color")

                # Troncal info
                troncal = route.get("troncal") or {}
                troncal_id = troncal.get("id")
                troncal_name = troncal.get("nombre")
                troncal_zone = troncal.get("zona")

                # Horarios info
                horarios = route.get("horarios") or []
                sched_mon_sat = ""
                sched_sun_hol = ""
                for h in horarios:
                    tipo_dia = h.get("tipoDia")
                    start = h.get("inicio")
                    end = h.get("fin")
                    if tipo_dia == "L-S":
                        sched_mon_sat = f"{start} - {end}"
                    elif tipo_dia == "D-F":
                        sched_sun_hol = f"{start} - {end}"

                safe_print(
                    f"[{idx}/{total_routes}] Route: {route_code} - {route_name}..."
                )

                # Fetch stations (paraderos)
                paraderos_url = (
                    f"{BASE_URL}/api/v1/rutas/{route_id}/{route_code}/paraderos"
                )
                paraderos = make_request(paraderos_url)

                if not paraderos:
                    safe_print(
                        f"  Warning: No stations found or error for Route {route_code}"
                    )
                    writer.writerow(
                        [
                            route_id,
                            route_code,
                            route_name,
                            route_color,
                            troncal_id,
                            troncal_name,
                            troncal_zone,
                            sched_mon_sat,
                            sched_sun_hol,
                            "",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )
                    continue

                for seq, stop in enumerate(paraderos, start=1):
                    stop_id = stop.get("id")
                    stop_code = stop.get("codigo")
                    stop_name = stop.get("nombre")
                    stop_address = stop.get("direccion")

                    writer.writerow(
                        [
                            route_id,
                            route_code,
                            route_name,
                            route_color,
                            troncal_id,
                            troncal_name,
                            troncal_zone,
                            sched_mon_sat,
                            sched_sun_hol,
                            seq,
                            stop_id,
                            stop_code,
                            stop_name,
                            stop_address,
                        ]
                    )

                time.sleep(sleep_delay)

        safe_print(f"✅ Extraction completed! CSV saved at: {output_file}")

    except Exception as e:
        safe_print(f"Error writing to CSV: {e}")
        raise


if __name__ == "__main__":
    # Test step
    params_path = Path(__file__).parent.parent / "params.json"
    if params_path.exists():
        with open(params_path) as f:
            params_dict = json.load(f)
        run(params_dict)
