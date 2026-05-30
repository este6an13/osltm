# TransMilenio Trunk Routes, Stations, and Frequencies Dataset

This directory contains structured, high-fidelity datasets of all active trunk routes, their sequential station stops, operational frequencies, and timetables within the **TransMilenio** mass transit system of Bogotá, Colombia.

The datasets were politely extracted and compiled from the public TransMilenio routes search API and the Tullave operational portal on **May 30, 2026**. They are formatted and documented for urban planning, transit analysis, civic-tech, and accessibility research.

---

## 📁 Dataset Overview

The dataset is divided into two primary complementary CSV files:
1. **[transmilenio_routes_stations.csv](transmilenio_routes_stations.csv)**: Detailed sequential mapping of all trunk lines showing every station stop in order of travel.
2. **[transmilenio_frequencies.csv](transmilenio_frequencies.csv)**: Route-level operational frequencies (headways) for peak and off-peak hours, operating schedules, and accessibility features.

---

## 📄 1. Sequential Routes & Stations Dataset

* **File Name:** [transmilenio_routes_stations.csv](transmilenio_routes_stations.csv)
* **Format:** Comma-Separated Values (CSV), UTF-8 encoded
* **Row Count:** 3,102 records (each representing a sequential stop along a route)

### 📋 Schema & Column Reference

| Column Name | Data Type | Description | Example |
| :--- | :---: | :--- | :--- |
| `route_id` | Integer | Internal database ID of the route. | `684` |
| `route_code` | String | Official alphanumeric route display code. | `1`, `B12`, `H21` |
| `route_name` | String | Terminal destination name of the route. | `Portal Eldorado` |
| `route_color` | String | Hex color code representing the route service. | `#D5B079` |
| `troncal_id` | Integer | Internal database ID of the corresponding trunk line corridor (*troncal*). | `10` |
| `troncal_name` | String | Display name of the trunk line corridor. | `Calle 26` |
| `troncal_zone` | String | Zone identifier letter (e.g., A, B, C, D, F, G, H, J, K, L, M). | `K` |
| `schedule_mon_sat` | String | Operating hours for Monday through Saturday. | `4:30 AM - 11:00 PM` |
| `schedule_sun_hol` | String | Operating hours for Sundays and Holidays. | `4:30 AM - 10:00 PM` |
| `station_sequence`| Integer | 1-based sequential order of the stop along the route's path. | `1`, `2`, `3` |
| `station_id` | Integer | Internal database ID of the stop. | `6111` |
| `station_code` | String | Official alphanumeric stop code. | `TM0122` |
| `station_name` | String | Name of the station stop. | `Universidades CityU` |
| `station_address` | String | Physical intersection or address location of the stop. | `KR 3 - CL 22` |

---

## 📄 2. Routes Frequencies & Schedules Dataset

* **File Name:** [transmilenio_frequencies.csv](transmilenio_frequencies.csv)
* **Format:** Comma-Separated Values (CSV), UTF-8 encoded
* **Row Count:** 87 records (representing outbound, inbound, or directional trunk routes)

### 📋 Schema & Column Reference

| Column Name | Data Type | Description | Example |
| :--- | :---: | :--- | :--- |
| `line_id` | Integer | Internal database ID of the route on the Tullave portal. | `10036` |
| `type` | String | Service category. Always `TRONCAL` (trunk) for this dataset. | `TRONCAL` |
| `name` | String | Official alphanumeric route display code (matches `route_code`). | `G45`, `M47`, `1`, `B10` |
| `destination` | String | Terminal destination of the route/direction. | `San Mateo` |
| `frequency_peak` | String | Average headway interval during peak hours (rush hours). | `4 min` |
| `frequency_offpeak`| String | Average headway interval during off-peak hours (valley periods). | `4 min` |
| `operation_days` | String | Calendar days of the week when the service operates. | `Lunes - Domingo` |
| `schedule` | String | Detailed span of service (operating hours) for this specific direction. | `Desde: 05:00:00 Hasta: 23:00:00` |
| `accessible` | String | Availability of wheelchair/reduced mobility boarding access. | `SÍ` |

> [!NOTE]
> - **Bidirectional & Numerical Routes:** Single digit numerical routes (e.g., `1`, `2`, `4`, `7`) run bidirectionally and map to two entries in the frequencies dataset (one for each terminal direction).
> - **Directional & Modern Routes:** Modern alphanumeric routes (e.g., `B13`, `H13`) are directional by design, where the prefix letter represents the destination zone. Each prefix has its own single direction line record.
> - **Missing Data Placeholders:** Placeholder values `---` indicate instances where the provider's API returned empty records at extraction time.

---

## 🏛️ Attribution & Reuse Policy

* **Sources:**
  1. TRANSMILENIO S.A. Operational Public Services Search API (`buscador-rutas.transmilenio.gov.co`).
  2. Tullave Plus Frequencies & Timetables Portal (`frecuencias.tullaveplus.gov.co`).
* **Open Government Framework:** This data consists of factual public utilities information. Under Colombia's **Law of Transparency and Access to Public Information** (Law 1712 of 2014) and the **National Open Data Initiative** (*Datos Abiertos Colombia*), operational facts, routes, timetables, and public utility transit data are freely reusable for academic, research, civic, or commercial software development.
* **License Recommendation:** Factual transit schedules, coordinates, and operational frequencies are public domain facts.

---
*Disclaimer: This repository is independent and not officially affiliated, endorsed, or sponsored by TRANSMILENIO S.A. or Tullave Plus.*
