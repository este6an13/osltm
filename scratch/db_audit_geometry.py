import pandas as pd
from pathlib import Path

nodes_path = Path("d:/dequi/repositories/osltm/data/geometry/output/nodes.csv")
routes_path = Path("d:/dequi/repositories/osltm/data/routes/transmilenio_routes_stations.csv")

def clean_id(val):
    try:
        if pd.isnull(val):
            return ""
        return str(int(float(val))).strip().zfill(5)
    except Exception:
        return str(val).strip().zfill(5)

print("--- AUDITING NODES.CSV ---")
df_nodes = pd.read_csv(nodes_path)
print(f"Total nodes: {len(df_nodes)}")

print("\n--- AUDITING ROUTES CSV ---")
df_routes = pd.read_csv(routes_path)
print(f"Total rows: {len(df_routes)}")

# Check mapping using clean_id
df_nodes["id_str"] = df_nodes["id"].apply(clean_id)
df_routes["sid_str"] = df_routes["station_id"].apply(clean_id)

matched_ids = set(df_nodes["id_str"]).intersection(set(df_routes["sid_str"]))
print(f"\nUnique node IDs in nodes.csv: {df_nodes['id_str'].nunique()}")
print(f"Unique station IDs in routes CSV: {df_routes['sid_str'].nunique()}")
print(f"Matched unique IDs: {len(matched_ids)}")

# Let's search for "B12" route stations in routes CSV and see how many match in nodes
print("\n--- B12 ROUTE DETAIL ---")
b12_df = df_routes[df_routes["route_code"] == "B12"]
print(f"Total B12 rows in routes CSV: {len(b12_df)}")
for _, row in b12_df.iterrows():
    sid = row["sid_str"]
    in_nodes = sid in set(df_nodes["id_str"])
    node_name = df_nodes[df_nodes["id_str"] == sid].iloc[0]["name"] if in_nodes else "NOT FOUND"
    print(f"Seq: {row['station_sequence']} | Route Station ID: {sid} ({row['station_name']}) | In Nodes: {in_nodes} ({node_name})")
