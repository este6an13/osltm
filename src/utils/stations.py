def extract_station_info(station_field: str):
    """
    Extract station code and name from the 'Estacion' column.
    Example: "(02202)Calle 127 - L Oreal Paris" -> ("02202", "Calle 127 - L Oreal Paris")
    """
    if not station_field or "(" not in station_field or ")" not in station_field:
        return None, None
    code = station_field.split(")")[0].replace("(", "").strip()
    name = station_field.split(")")[1].strip()
    return code, name
