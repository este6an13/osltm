# TransMilenio Trunk Routes & Stations Dataset

This folder contains a structured, high-fidelity dataset of all active trunk routes and their sequential stations within the **TransMilenio** mass transit system of Bogotá, Colombia.

The data was politely extracted from the public TransMilenio routes microservice API on **May 30, 2026**. It is compiled and formatted for urban planning, civic-tech, and accessibility research.

---

## 📄 Dataset File
* **File name:** [transmilenio_routes_stations.csv](transmilenio_routes_stations.csv)
* **Format:** Comma-Separated Values (CSV), UTF-8 encoded
* **Row count:** 3,102 records (each representing a sequential stop along a route)

---

## 📋 CSV Schema & Column Reference

| Column Name | Data Type | Description | Example |
| :--- | :---: | :--- | :--- |
| `route_id` | Integer | Internal database ID of the route. | `684` |
| `route_code` | String | Official alphanumeric route display code. | `1`, `B12`, `H21` |
| `route_name` | String | Terminal destination name of the route. | `Portal Eldorado` |
| `route_color` | String | Hex color code representing the route service. | `#D5B079` |
| `troncal_id` | Integer | Internal database ID of the corresponding trunk line (*troncal*). | `10` |
| `troncal_name` | String | Display name of the trunk line corridor. | `Calle 26` |
| `troncal_zone` | String | Zone identifier letter. | `K` |
| `schedule_mon_sat` | String | Operating hours for Monday through Saturday. | `4:30 AM - 11:00 PM` |
| `schedule_sun_hol` | String | Operating hours for Sundays and Holidays. | `4:30 AM - 10:00 PM` |
| `station_sequence`| Integer | 1-based order of the stop along the route's path. | `1`, `2`, `3` |
| `station_id` | Integer | Internal database ID of the stop. | `6111` |
| `station_code` | String | Official alphanumeric stop code. | `TM0122` |
| `station_name` | String | Name of the station stop. | `Universidades CityU` |
| `station_address` | String | Physical intersection or address location of the stop. | `KR 3 - CL 22` |

---

## 🏛️ Attribution & Reuse Policy
* **Source:** TRANSMILENIO S.A. Operational Public Services Search API (`buscador-rutas.transmilenio.gov.co`).
* **Open Government Framework:** This data consists of factual public utilities information. Under Colombia's **Law of Transparency and Access to Public Information** (Law 1712 of 2014) and the **National Open Data Initiative** (*Datos Abiertos Colombia*), Operational facts and public utility transit data are freely reusable for academic, civic, or commercial software development.
* **License Recommendation:** Factual transit schedules and addresses are considered public domain facts.

*Disclaimer: This repository is independent and not officially affiliated with TRANSMILENIO S.A.*
