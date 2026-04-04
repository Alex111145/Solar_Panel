"""
pvgclient.py
===============
Recupera dati di irradiazione solare inviando le coordinate GPS
(Latitudine e Longitudine) del centroide dell'ortomosaico direttamente
alle API ufficiali di PVGIS (JRC Europa).

Nessuna API key richiesta — servizi pubblici gratuiti.
"""

import json
import urllib.request
import urllib.parse
import urllib.error

def _call_pvgis(lat: float, lon: float) -> dict:
    """
    Chiama PVGIS v5.2 con retry automatico (3 tentativi, backoff).
    Prova prima v5.2, poi v5.1 come fallback.
    """
    params = urllib.parse.urlencode({
        "lat":           lat,
        "lon":           lon,
        "peakpower":     1,
        "loss":          0,
        "outputformat":  "json",
        "mountingplace": "free",
        "optimalangles": 1,
    })

    endpoints = [
        f"https://re.jrc.ec.europa.eu/api/v5_2/PVcalc?{params}",
        f"https://re.jrc.ec.europa.eu/api/v5_1/PVcalc?{params}",
    ]

    last_error = None
    for endpoint in endpoints:
        req = urllib.request.Request(endpoint, headers={"User-Agent": "SolarPanelAnalysis/1.0"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                raise ValueError(f"PVGIS API error {e.code}: {e.reason}")
            except Exception as e:
                last_error = e
                wait = 2 ** attempt   # 1s, 2s, 4s
                print(f"  ⚠️  Tentativo {attempt+1}/3 fallito ({e}) — riprovo in {wait}s...")
                import time
                time.sleep(wait)

    raise ValueError(f"PVGIS non raggiungibile dopo 6 tentativi: {last_error}")

def _parse_pvgis(data: dict, lat: float) -> dict:
    """
    Estrae i dati di irradiazione e calcola ESH e Giorni Utili.
    """
    try:
        monthly = data["outputs"]["monthly"]["fixed"]
        irr_mensile = [m["H(i)_m"] for m in monthly]   # kWh/m²/mese
    except KeyError:
        irr_mensile = [0.0] * 12

    irr_annua = sum(irr_mensile)
    
    # ESH = ore equivalenti = irradiazione annua / 1000 W/m²
    esh = irr_annua / 1000.0

    # Giorni utili: mesi con media giornaliera > 3 kWh/m²/gg → ~giorni/anno
    giorni_per_mese = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    giorni_utili = sum(
        gg for m_kwh, gg in zip(irr_mensile, giorni_per_mese)
        if (m_kwh / gg) >= 3.0
    )

    # Angolo ottimale
    try:
        angolo_opt = data["inputs"]["mounting_system"]["fixed"]["slope"]["value"]
    except KeyError:
        angolo_opt = round(abs(lat) * 0.76 + 3.1, 1)  # approssimazione

    return {
        "irr_annua":    round(irr_annua, 1),
        "esh":          round(esh, 2),
        "giorni_utili": int(giorni_utili),
        "irr_mensile":  [round(v, 1) for v in irr_mensile],
        "angolo_opt":   angolo_opt,
    }

def get_pvgis_data(lat: float, lon: float) -> dict:
    """
    Recupera i dati PVGIS partendo direttamente dalle coordinate assolute.
    """
    print(f"  ☀️  Recupero dati PVGIS per le coordinate: Lat {lat:.6f}, Lon {lon:.6f} ...")
    raw = _call_pvgis(lat, lon)
    result = _parse_pvgis(raw, lat)
    result.update({"lat": lat, "lon": lon})

    print(f"     → Irradiazione annua : {result['irr_annua']} kWh/m²/anno")
    print(f"     → ESH                : {result['esh']} h/anno")
    print(f"     → Giorni utili       : {result['giorni_utili']} giorni/anno")
    print(f"     → Angolo ottimale    : {result['angolo_opt']}°")

    return result

if __name__ == "__main__":
    import sys
    print("Test standalone su un punto di esempio (Milano Centro)...")
    try:
        d = get_pvgis_data(45.4642, 9.1900)
        print(f"\n{'='*50}")
        print(f"  Coordinate    : {d['lat']:.4f}, {d['lon']:.4f}")
        print(f"  Irr. annua    : {d['irr_annua']} kWh/m²/anno")
        print(f"  ESH           : {d['esh']} h/anno")
        print(f"  Giorni utili  : {d['giorni_utili']} giorni")
        print(f"  Angolo opt.   : {d['angolo_opt']}°")
        print(f"{'='*50}")
    except ValueError as e:
        print(f"❌ {e}")
