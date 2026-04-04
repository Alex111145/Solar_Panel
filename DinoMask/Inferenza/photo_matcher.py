"""
photo_matcher.py
================
Trova la foto drone originale (RJPEG DJI) che copre un pannello solare
rilevato nell'ortomosaico, dato:
  - le coordinate GPS del pannello (lat, lon WGS84)
  - la cartella contenente le foto raw del drone

Strategia di matching:
  1. Legge le coordinate GPS EXIF di ogni foto nella cartella drone
  2. Stima il footprint (area coperta) dalla foto in base ad altitudine e FOV
  3. Restituisce il/i file che contengono il punto GPS del pannello nel footprint

Funzioni pubbliche:
  - load_drone_photos_index(drone_folder)  → lista dicts {path, lat, lon, footprint}
  - find_photos_for_panel(lat, lon, index) → lista path file
  - find_best_photo(lat, lon, index)       → path file con centro più vicino

Uso:
    from photo_matcher import load_drone_photos_index, find_best_photo
    index = load_drone_photos_index("/path/foto_drone")
    photo = find_best_photo(panel_lat, panel_lon, index)
"""

import os
import json
import math
import subprocess
from typing import Optional

# FOV fallback Zenmuse H30T termico
H30T_HFOV_DEG = 40.6
H30T_VFOV_DEG = 32.1
H30T_ALT_DEFAULT = 30.0   # altitudine default se non leggibile


# ==============================================================================
# LETTURA GPS EXIF FOTO DRONE
# ==============================================================================

def _parse_xmp_dji(photo_path: str) -> Optional[dict]:
    """
    Legge i metadati GPS DJI dal blocco XMP interno al JPG/RJPEG.
    DJI salva le coordinate nei tag drone-dji:GpsLatitude/GpsLongitude.
    Funziona anche quando exiftool non trova i campi standard.
    """
    try:
        from PIL import Image
        img = Image.open(photo_path)
        xmp_bytes = img.info.get("xmp", b"")
        if not xmp_bytes:
            return None
        xmp = xmp_bytes.decode("utf-8", errors="ignore")

        import re
        def _get(tag):
            m = re.search(rf'drone-dji:{tag}="([^"]+)"', xmp)
            return m.group(1) if m else None

        lat_s = _get("GpsLatitude")
        lon_s = _get("GpsLongitude")
        if not lat_s or not lon_s:
            return None

        lat = float(lat_s)
        lon = float(lon_s)
        alt = abs(float(_get("RelativeAltitude") or H30T_ALT_DEFAULT))

        return {
            "lat":       lat,
            "lon":       lon,
            "alt_m":     alt,
            "fov_h_deg": H30T_HFOV_DEG,
            "fov_v_deg": H30T_VFOV_DEG,
        }
    except Exception:
        return None


def _dms_to_decimal(dms, ref: str) -> float:
    """Converte GPS DMS (tuple PIL) in gradi decimali."""
    d, m, s = dms
    dec = float(d) + float(m) / 60.0 + float(s) / 3600.0
    if ref in ('S', 'W'):
        dec = -dec
    return dec


def _read_gps_exif(photo_path: str) -> Optional[dict]:
    """
    Legge GPS e metadati di volo dall'EXIF tramite PIL (senza exiftool).
    Strategia:
      1. XMP DJI (drone-dji:GpsLatitude) — presente in alcuni RJPEG
      2. EXIF standard GPSInfo via PIL._getexif() — DMS → decimale
    """
    # Strategia 1: XMP DJI
    result_xmp = _parse_xmp_dji(photo_path)
    if result_xmp:
        return result_xmp

    # Strategia 2: EXIF GPS via PIL (nessuna dipendenza esterna)
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS

        img  = Image.open(photo_path)
        exif = img._getexif()
        if not exif:
            return None

        # Trova il blocco GPSInfo
        gps_info = None
        for tag_id, val in exif.items():
            if TAGS.get(tag_id) == "GPSInfo":
                gps_info = {GPSTAGS.get(k, k): v for k, v in val.items()}
                break

        if not gps_info:
            return None

        lat_dms = gps_info.get("GPSLatitude")
        lat_ref = gps_info.get("GPSLatitudeRef", "N")
        lon_dms = gps_info.get("GPSLongitude")
        lon_ref = gps_info.get("GPSLongitudeRef", "E")

        if not lat_dms or not lon_dms:
            return None

        lat = _dms_to_decimal(lat_dms, lat_ref)
        lon = _dms_to_decimal(lon_dms, lon_ref)

        # Altitudine: usa GPSAltitude (ASL) come approssimazione AGL
        # DJI vola tipicamente a 20-50m AGL — se ASL > 100m clip a default
        alt_asl = float(gps_info.get("GPSAltitude", H30T_ALT_DEFAULT))
        alt = alt_asl if alt_asl <= 80.0 else H30T_ALT_DEFAULT

        return {
            "lat":       lat,
            "lon":       lon,
            "alt_m":     alt,
            "fov_h_deg": H30T_HFOV_DEG,
            "fov_v_deg": H30T_VFOV_DEG,
        }

    except Exception:
        return None


def _compute_footprint(gps: dict) -> dict:
    """
    Calcola il footprint geografico (bounding box lat/lon) della foto.

    Footprint orizzontale:
        W_ground = 2 * alt * tan(HFOV/2)
        H_ground = 2 * alt * tan(VFOV/2)

    Nota: assume volo nadirale (nessun rollio/beccheggio).
    I GSD termici DJI variano tra ~0.5 cm/px (5 m) e ~10 cm/px (100 m).
    """
    alt   = gps["alt_m"]
    fov_h = math.radians(gps["fov_h_deg"])
    fov_v = math.radians(gps["fov_v_deg"])

    w_ground_m = 2.0 * alt * math.tan(fov_h / 2.0)   # metri orizzontali
    h_ground_m = 2.0 * alt * math.tan(fov_v / 2.0)   # metri verticali

    # Converti m → gradi (approssimazione piana, sufficiente per distanze <1 km)
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(gps["lat"])))

    half_lat = (h_ground_m / 2.0) * deg_per_m_lat
    half_lon = (w_ground_m / 2.0) * deg_per_m_lon

    return {
        "lat_min": gps["lat"] - half_lat,
        "lat_max": gps["lat"] + half_lat,
        "lon_min": gps["lon"] - half_lon,
        "lon_max": gps["lon"] + half_lon,
        "w_m":     w_ground_m,
        "h_m":     h_ground_m,
    }


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Distanza in metri tra due punti GPS (Haversine)."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ==============================================================================
# INDICE FOTO DRONE
# ==============================================================================

def load_drone_photos_index(drone_folder: str,
                             extensions: tuple = (".jpg", ".jpeg", ".tiff", ".tif")
                             ) -> list[dict]:
    """
    Scansiona la cartella drone e costruisce l'indice di tutte le foto con GPS.

    Args:
        drone_folder : cartella contenente i file RJPEG DJI
        extensions   : estensioni da considerare

    Returns:
        lista di dict, ognuno con:
          path, lat, lon, alt_m, fov_h_deg, fov_v_deg,
          footprint {lat_min, lat_max, lon_min, lon_max, w_m, h_m}

    Note:
        Il risultato viene messo in cache (variabile globale) per evitare
        letture EXIF ripetute sullo stesso folder nella stessa sessione.
    """
    # Cache per non rileggere EXIF ogni volta
    cache_key = os.path.abspath(drone_folder)
    if cache_key in _photo_index_cache:
        return _photo_index_cache[cache_key]

    if not os.path.isdir(drone_folder):
        print(f"  [photo_matcher] Cartella non trovata: {drone_folder}")
        return []

    files = [
        os.path.join(drone_folder, f)
        for f in sorted(os.listdir(drone_folder))
        if os.path.splitext(f.lower())[1] in extensions
    ]

    print(f"  [photo_matcher] Scansione {len(files)} foto in '{drone_folder}'...")

    index = []
    skipped = 0
    for i, fpath in enumerate(files):
        gps = _read_gps_exif(fpath)
        if gps is None:
            skipped += 1
            continue
        footprint = _compute_footprint(gps)
        index.append({
            "path":      fpath,
            "lat":       gps["lat"],
            "lon":       gps["lon"],
            "alt_m":     gps["alt_m"],
            "fov_h_deg": gps["fov_h_deg"],
            "fov_v_deg": gps["fov_v_deg"],
            "footprint": footprint,
        })
        if (i + 1) % 50 == 0:
            print(f"  [photo_matcher]   ... indicizzate {i+1}/{len(files)}")

    if skipped:
        print(f"  [photo_matcher] {skipped} foto senza GPS saltate")
    print(f"  [photo_matcher] Indice: {len(index)} foto con GPS")

    _photo_index_cache[cache_key] = index
    return index


_photo_index_cache: dict[str, list] = {}


# ==============================================================================
# RICERCA FOTO PER PANNELLO
# ==============================================================================

def find_photos_for_panel(lat: float, lon: float,
                           index: list[dict],
                           margin_factor: float = 0.10
                           ) -> list[dict]:
    """
    Trova tutte le foto il cui footprint contiene il punto (lat, lon).

    Args:
        lat, lon       : coordinate GPS del pannello
        index          : lista da load_drone_photos_index()
        margin_factor  : espande il footprint del X% per tolleranza bordi

    Returns:
        Lista di entry dell'index che coprono il punto, ordinate per distanza
        crescente dal centro della foto (la prima è la migliore).
    """
    candidates = []
    for entry in index:
        fp = entry["footprint"]
        margin_lat = (fp["lat_max"] - fp["lat_min"]) * margin_factor
        margin_lon = (fp["lon_max"] - fp["lon_min"]) * margin_factor

        if (fp["lat_min"] - margin_lat <= lat <= fp["lat_max"] + margin_lat and
                fp["lon_min"] - margin_lon <= lon <= fp["lon_max"] + margin_lon):
            dist = _haversine(lat, lon, entry["lat"], entry["lon"])
            candidates.append((dist, entry))

    candidates.sort(key=lambda x: x[0])
    return [e for _, e in candidates]


def find_best_photo(lat: float, lon: float,
                    index: list[dict],
                    margin_factor: float = 0.10
                    ) -> Optional[str]:
    """
    Restituisce il path della foto drone più adatta per il pannello (lat, lon).

    Preferisce la foto il cui centro è più vicino al pannello (migliore
    angolo nadirale, massima risoluzione nel punto).

    Returns:
        path del file RJPEG, oppure None se nessuna foto copre il punto.
    """
    matches = find_photos_for_panel(lat, lon, index, margin_factor)
    if not matches:
        return None
    return matches[0]["path"]


def panel_pixel_in_photo(panel_lat: float, panel_lon: float,
                          photo_entry: dict) -> tuple[int, int]:
    """
    Calcola le coordinate pixel (col, row) del punto GPS del pannello
    all'interno della foto drone.

    Utile per estrarre il crop esatto da passare a thermal_extractor.

    Returns:
        (col, row) coordinate pixel nella foto
    """
    fp   = photo_entry["footprint"]
    fov_h = photo_entry["fov_h_deg"]
    fov_v = photo_entry["fov_v_deg"]
    alt   = photo_entry["alt_m"]

    # Ground footprint in m
    w_m = fp["w_m"]
    h_m = fp["h_m"]

    # Calcola GSD dalla foto
    import math
    # Stima dimensioni sensore: H30T = 640×512
    W_PX = 640
    H_PX = 512
    gsd_x = w_m / W_PX
    gsd_y = h_m / H_PX

    # Offset del pannello dal centro della foto
    deg_per_m_lat = 1.0 / 111320.0
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(photo_entry["lat"])))

    delta_lat_m = (panel_lat - photo_entry["lat"]) / deg_per_m_lat
    delta_lon_m = (panel_lon - photo_entry["lon"]) / deg_per_m_lon

    # (0,0) = top-left; centro = (W_PX/2, H_PX/2)
    col = int(W_PX / 2 + delta_lon_m / gsd_x)
    row = int(H_PX / 2 - delta_lat_m / gsd_y)  # lat cresce verso nord (→ row decresce)

    col = max(0, min(col, W_PX - 1))
    row = max(0, min(row, H_PX - 1))

    return col, row


# ==============================================================================
# MAIN — test standalone
# ==============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python photo_matcher.py <cartella_drone> [lat] [lon]")
        print("Esempio: python photo_matcher.py ./foto_drone 45.1234 9.5678")
        sys.exit(0)

    folder = sys.argv[1]

    # Solo scansione se lat/lon non fornite
    if len(sys.argv) < 4:
        print(f"\n{'='*55}")
        print(f"  SCANSIONE CARTELLA: {folder}")
        print(f"{'='*55}")
        idx = load_drone_photos_index(folder)
        if not idx:
            print("  ❌ Nessuna foto con GPS trovata")
            sys.exit(1)
        print(f"\n  Prime 5 foto indicizzate:")
        for e in idx[:5]:
            print(f"  {os.path.basename(e['path'])}")
            print(f"    GPS: {e['lat']:.6f}, {e['lon']:.6f}  |  Alt: {e['alt_m']:.1f} m")
            fp = e['footprint']
            print(f"    Footprint: {fp['w_m']:.1f} m × {fp['h_m']:.1f} m")
        print(f"\n  Riesegui con: python photo_matcher.py {folder} {idx[0]['lat']:.6f} {idx[0]['lon']:.6f}")
        print(f"{'='*55}\n")
        sys.exit(0)

    lat = float(sys.argv[2])
    lon = float(sys.argv[3])

    print(f"\n{'='*55}")
    print(f"  Ricerca foto per pannello GPS ({lat:.6f}, {lon:.6f})")
    print(f"{'='*55}")

    idx = load_drone_photos_index(folder)
    if not idx:
        print("  ❌ Nessuna foto con GPS trovata")
        sys.exit(1)

    matches = find_photos_for_panel(lat, lon, idx)
    if not matches:
        print(f"  ⚠️  Nessuna foto copre il punto ({lat:.6f}, {lon:.6f})")
    else:
        print(f"  ✅ Trovate {len(matches)} foto che coprono il punto:")
        for i, m in enumerate(matches[:5]):
            dist = _haversine(lat, lon, m["lat"], m["lon"])
            col, row = panel_pixel_in_photo(lat, lon, m)
            print(f"  [{i+1}] {os.path.basename(m['path'])}")
            print(f"       Centro foto: ({m['lat']:.6f}, {m['lon']:.6f})  dist: {dist:.1f} m")
            print(f"       Pannello in foto: col={col} row={row}")
    print(f"{'='*55}\n")
