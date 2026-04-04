"""
thermal_extractor.py
====================
Estrae dati termici da foto RJPEG DJI (Matrice 4DT / Zenmuse H30T).

Funzioni principali:
  - get_temperature_matrix(rjpeg_path)   → np.ndarray 2D di temperature (°C)
  - get_exif_metadata(rjpeg_path)        → dict con T_amb, altitudine, FOV, ecc.
  - get_gsd(rjpeg_path)                  → GSD in m/pixel (Ground Sample Distance)
  - get_ambient_temperature(rjpeg_path)  → T_amb in °C dai metadati EXIF
  - extract_panel_temperature(rjpeg_path, mask_pixels, mode) → temperatura pannello
  - extract_from_patch_region(rjpeg_path, patch_offset_x, patch_offset_y,
                               mask_pixels_in_patch, mode) → temp dal crop

Dipendenze:
  - dji_thermal_sdk (libreria DJI, .so / .pyd)
  - Fallback: subprocess con dji_irp (CLI tool)
  - numpy, Pillow

Uso:
    from thermal_extractor import get_gsd, get_ambient_temperature, extract_panel_temperature
    t_amb = get_ambient_temperature("foto_drone/DJI_0001.jpg")
    gsd   = get_gsd("foto_drone/DJI_0001.jpg")
    t_pan = extract_panel_temperature("foto_drone/DJI_0001.jpg", mask_pixels, mode="mean")
"""

import os
import json
import struct
import subprocess
import tempfile
import numpy as np

# ==============================================================================
# COSTANTI TERMICHE DJI
# ==============================================================================

EMISSIVITY_GLASS   = 0.95          # pannelli FV → vetro
REFLECTED_TEMP_C   = 25.0          # temperatura riflessa di default (°C)
ATMOSPHERIC_TEMP_C = 25.0          # temperatura atmosferica di default (°C)
HUMIDITY           = 0.70          # umidità relativa default (70%)

# FOV e risoluzione nativa sensore termico DJI M4DT
# Il sensore produce 640×512 raw — il 960×768 è upscaled visuale
H30T_HFOV_DEG  = 40.6
H30T_VFOV_DEG  = 32.1
H30T_WIDTH_PX  = 640   # risoluzione nativa SDK dirp_get_rjpeg_resolution
H30T_HEIGHT_PX = 512

# ==============================================================================
# IMPORT DJI THERMAL SDK (con fallback graceful)
# ==============================================================================

_SDK_AVAILABLE = False
_DJI_IRP_CLI   = None

# Cartella base dello script
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sottocartella platform-specific: sdk/linux  oppure  sdk/windows
_PLATFORM     = "windows" if os.name == "nt" else "linux"
_LIBS_DIR     = os.path.join(_BASE_DIR, "sdk", _PLATFORM)

# Estensione librerie native per piattaforma
_LIB_EXT      = ".dll" if os.name == "nt" else ".so"
_IRP_EXE      = "dji_irp.exe" if os.name == "nt" else "dji_irp"
_LIBDIRP_NAME = "libdirp" + _LIB_EXT


def _init_dji_sdk():
    """
    Carica libdirp + dipendenze DJI dalla cartella sdk/<platform>/.
    Ritorna True se l'init ha successo.
    """
    import ctypes

    libdirp_path = os.path.join(_LIBS_DIR, _LIBDIRP_NAME)
    if not os.path.exists(libdirp_path):
        return False

    # Dipendenze Linux (.so) — su Windows le .dll si caricano automaticamente
    if os.name != "nt":
        for dep in [
            "libMicroJPEG_Release_x64.so", "libMicroIA_Release_x64.so",
            "libMicroTA_Release_x64.so",   "libv_cirp.so", "libv_dirp.so",
            "libv_girp.so", "libv_hirp.so", "libv_iirp.so",
        ]:
            p = os.path.join(_LIBS_DIR, dep)
            if os.path.exists(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

    try:
        from dji_thermal_sdk.dji_sdk import dji_init
        dji_init(libdirp_path, osname=_PLATFORM)
        return True
    except Exception:
        return False


try:
    import dji_thermal_sdk as _dji_mod
    _SDK_AVAILABLE = _init_dji_sdk()
    if _SDK_AVAILABLE:
        print(f"[DJI thermal] ✅ SDK Python attivo  ({_LIBS_DIR})")
    else:
        print(f"[DJI thermal] ❌ SDK init fallito — {_LIBDIRP_NAME} non trovato in {_LIBS_DIR}")
except ImportError:
    print("[DJI thermal] ❌ dji_thermal_sdk non installato  (pip install dji-thermal-sdk)")

if not _SDK_AVAILABLE:
    for _candidate in [
        _IRP_EXE,
        os.path.join(_LIBS_DIR, _IRP_EXE),   # sdk/linux/dji_irp  o  sdk/windows/dji_irp.exe
        "/usr/local/bin/dji_irp",
    ]:
        try:
            if os.path.isfile(_candidate) or subprocess.run(
                ["where" if os.name == "nt" else "which", _candidate],
                capture_output=True
            ).returncode == 0:
                _DJI_IRP_CLI = _candidate
                print(f"[DJI thermal] ⚠️  Fallback CLI: {_DJI_IRP_CLI}")
                break
        except Exception:
            pass
    if not _DJI_IRP_CLI:
        print("[DJI thermal] ⚠️  Fallback raw parsing (meno accurato, solo H20T/H30T/XT2)")


# ==============================================================================
# LETTURA EXIF / METADATI XMPRC
# ==============================================================================

def get_exif_metadata(rjpeg_path: str) -> dict:
    """
    Legge i metadati EXIF e XMP-RC di un file RJPEG DJI.

    Campi estratti (con fallback 0/None se assenti):
      altitude_m       : quota AGL in metri
      t_amb_c          : temperatura ambiente (AtmosphericTemperature)
      humidity         : umidità relativa (0-1)
      emissivity       : emissività impostata nella camera
      reflected_temp_c : temperatura riflessa
      fov_h_deg        : FOV orizzontale (gradi)
      fov_v_deg        : FOV verticale (gradi)
      image_width      : larghezza immagine in pixel
      image_height     : altezza immagine in pixel

    Richiede exiftool installato nel sistema.
    Fallback: valori di default costanti se exiftool non disponibile.
    """
    meta = {
        "altitude_m":       0.0,
        "t_amb_c":          ATMOSPHERIC_TEMP_C,
        "humidity":         HUMIDITY,
        "emissivity":       EMISSIVITY_GLASS,
        "reflected_temp_c": REFLECTED_TEMP_C,
        "fov_h_deg":        H30T_HFOV_DEG,
        "fov_v_deg":        H30T_VFOV_DEG,
        "image_width":      H30T_WIDTH_PX,
        "image_height":     H30T_HEIGHT_PX,
    }

    try:
        # Prima prova: XMP DJI diretto + dimensioni reali frame termico
        try:
            from PIL import Image as _PILImage
            import re as _re
            _img = _PILImage.open(rjpeg_path)

            # Leggi dimensioni dal frame termico reale (frame 1 nei file MPO DJI)
            try:
                _img.seek(1)
                meta["image_width"]  = _img.size[0]
                meta["image_height"] = _img.size[1]
                _img.seek(0)
            except Exception:
                pass

            _xmp = _img.info.get("xmp", b"").decode("utf-8", errors="ignore")
            if _xmp:
                def _xget(tag):
                    m = _re.search(rf'drone-dji:{tag}="([^"]+)"', _xmp)
                    return m.group(1) if m else None

                _alt = _xget("RelativeAltitude")
                if _alt:
                    meta["altitude_m"] = abs(float(_alt))

                _st = _xget("SensorTemperature")
                if _st and float(_st) != 0.0:
                    meta["t_amb_c"] = float(_st)
        except Exception:
            pass

        result = subprocess.run(
            ["exiftool", "-j", "-n", rjpeg_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return meta

        data = json.loads(result.stdout)[0]

        # Altitudine AGL (solo se non già trovata da XMP)
        if meta["altitude_m"] == 0.0:
            for key in ["RelativeAltitude", "GPSAltitude"]:
                if key in data:
                    try:
                        meta["altitude_m"] = abs(float(data[key]))
                        break
                    except (ValueError, TypeError):
                        pass

        # Temperatura ambiente (XMP DJI)
        for key in ["AtmosphericTemperature", "AmbientTemperature",
                    "DJI:AtmosphericTemperature"]:
            if key in data:
                try:
                    meta["t_amb_c"] = float(data[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Umidità
        for key in ["RelativeHumidity", "AtmosphericHumidity"]:
            if key in data:
                try:
                    meta["humidity"] = float(data[key]) / 100.0 \
                        if float(data[key]) > 1.0 else float(data[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Emissività
        for key in ["Emissivity", "IRWindowEmissivity"]:
            if key in data:
                try:
                    meta["emissivity"] = float(data[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Temperatura riflessa
        for key in ["ReflectedApparentTemperature", "ReflectedTemperature"]:
            if key in data:
                try:
                    meta["reflected_temp_c"] = float(data[key])
                    break
                except (ValueError, TypeError):
                    pass

        # Dimensioni immagine
        if "ImageWidth" in data:
            meta["image_width"] = int(data["ImageWidth"])
        if "ImageHeight" in data:
            meta["image_height"] = int(data["ImageHeight"])

        # FOV (DJI salva DFOV, HFOV, VFOV in XMP)
        for key in ["HFOV", "CameraHFOV", "DJI:HFOV"]:
            if key in data:
                try:
                    meta["fov_h_deg"] = float(data[key])
                    break
                except (ValueError, TypeError):
                    pass

        for key in ["VFOV", "CameraVFOV", "DJI:VFOV"]:
            if key in data:
                try:
                    meta["fov_v_deg"] = float(data[key])
                    break
                except (ValueError, TypeError):
                    pass

    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError,
            IndexError, KeyError):
        pass  # usa valori di default

    return meta


def _get_tamb_from_openmeteo(lat: float, lon: float, utc_datetime: str) -> float | None:
    """
    Recupera la temperatura aria (°C) da Open-Meteo Historical API
    usando coordinate GPS e data/ora UTC del volo.

    Args:
        lat, lon      : coordinate GPS della foto
        utc_datetime  : stringa ISO8601 es. "2026-04-01T10:29:43.030356"

    Returns:
        temperatura in °C oppure None se fallisce
    """
    try:
        import urllib.request, urllib.parse, json, math

        # Estrai data e ora dalla stringa ISO
        dt_part = utc_datetime[:19]          # "2026-04-01T10:29:43"
        date    = dt_part[:10]               # "2026-04-01"
        hour    = int(dt_part[11:13])        # 10

        params = urllib.parse.urlencode({
            "latitude":        round(lat, 4),
            "longitude":       round(lon, 4),
            "start_date":      date,
            "end_date":        date,
            "hourly":          "temperature_2m",
            "timezone":        "UTC",
            "temperature_unit": "celsius",
        })
        url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "SolarPanelAnalysis/1.0"})

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        temps = data["hourly"]["temperature_2m"]   # 24 valori (ora 0-23)
        t_amb = temps[hour]
        return float(t_amb)

    except Exception as e:
        print(f"  [thermal] Open-Meteo fallito: {e}")
        return None


def get_ambient_temperature(rjpeg_path: str) -> float:
    """
    Restituisce la temperatura ambientale (°C):
      1. Open-Meteo API storica (data+ora+GPS dal XMP DJI)
      2. Fallback: 25.0°C
    """
    # Leggi lat/lon e UTC dal XMP DJI
    try:
        from PIL import Image
        import re
        img  = Image.open(rjpeg_path)
        xmp  = img.info.get("xmp", b"").decode("utf-8", errors="ignore")

        def _xget(tag):
            m = re.search(rf'drone-dji:{tag}="([^"]+)"', xmp)
            return m.group(1) if m else None

        lat_s = _xget("GpsLatitude")
        lon_s = _xget("GpsLongitude")
        utc_s = _xget("UTCAtExposure")

        if lat_s and lon_s and utc_s:
            lat = float(lat_s)
            lon = float(lon_s)
            print(f"  [T_amb] Richiesta Open-Meteo: lat={lat:.4f} lon={lon:.4f} UTC={utc_s[:19]}")
            t_amb = _get_tamb_from_openmeteo(lat, lon, utc_s)
            if t_amb is not None:
                print(f"  [T_amb] Temperatura aria da Open-Meteo: {t_amb:.1f} °C")
                return t_amb
    except Exception as e:
        print(f"  [T_amb] Errore lettura XMP: {e}")

    print(f"  [T_amb] Fallback: {ATMOSPHERIC_TEMP_C:.1f} °C (default)")
    return ATMOSPHERIC_TEMP_C


# ==============================================================================
# CALCOLO GSD
# ==============================================================================

def get_gsd(rjpeg_path: str) -> float:
    """
    Calcola il GSD (m/pixel) leggendo risoluzione e altitudine direttamente
    dall'SDK DJI (dati più affidabili del EXIF).

    Formula: GSD = 2 * alt * tan(HFOV/2) / sensor_width_px

    La risoluzione nativa del sensore termico DJI M4DT è 640×512 px
    (restituita da dirp_get_rjpeg_resolution).
    L'altitudine è letta dall'XMP DJI (RelativeAltitude, RTK).

    Returns:
        GSD in m/pixel. Fallback: 0.02895 m/px (640px, alt=24m, HFOV=40.6°)
    """
    import math

    # Leggi risoluzione reale del sensore dall'SDK
    sensor_width = H30T_WIDTH_PX   # default 640 (M4DT nativo)
    if _SDK_AVAILABLE:
        try:
            import ctypes
            from dji_thermal_sdk.dji_sdk import (
                dirp_create_from_rjpeg, dirp_get_rjpeg_resolution,
                dirp_destroy, DIRP_SUCCESS, dirp_resolution_t
            )
            with open(rjpeg_path, "rb") as f:
                data = f.read()
            buf    = ctypes.create_string_buffer(data)
            handle = ctypes.c_void_p()
            ret    = dirp_create_from_rjpeg(buf, ctypes.c_int32(len(data)),
                                             ctypes.byref(handle))
            if ret == DIRP_SUCCESS:
                res = dirp_resolution_t()
                dirp_get_rjpeg_resolution(handle, ctypes.byref(res))
                if res.width > 0:
                    sensor_width = res.width
                dirp_destroy(handle)
        except Exception:
            pass

    # Leggi altitudine AGL dall'XMP DJI (RTK, alta precisione)
    alt = 0.0
    try:
        from PIL import Image
        import re
        img = Image.open(rjpeg_path)
        xmp = img.info.get("xmp", b"").decode("utf-8", errors="ignore")
        m   = re.search(r'drone-dji:RelativeAltitude="([^"]+)"', xmp)
        if m:
            alt = abs(float(m.group(1)))
    except Exception:
        pass

    if alt < 0.5:
        alt = get_exif_metadata(rjpeg_path)["altitude_m"]

    fov_h_rad = math.radians(H30T_HFOV_DEG)
    if alt > 0.5 and sensor_width > 0:
        footprint_w = 2.0 * alt * math.tan(fov_h_rad / 2.0)
        gsd = footprint_w / sensor_width
        if 0.001 <= gsd <= 1.0:
            return round(gsd, 5)

    return 0.02895   # fallback: 640px, alt=24m, HFOV=40.6°


# ==============================================================================
# ESTRAZIONE MATRICE TEMPERATURE
# ==============================================================================

def get_temperature_matrix(rjpeg_path: str,
                            emissivity: float = EMISSIVITY_GLASS) -> np.ndarray:
    """
    Estrae la matrice 2D di temperature (°C) dall'immagine RJPEG DJI.

    Strategia (in ordine di priorità):
      1. dji_thermal_sdk Python (se disponibile)
      2. CLI dji_irp (se disponibile nel PATH)
      3. Parsing raw RJPEG: estrae il blob termico dai byte dell'APP4/APP3

    Returns:
        np.ndarray shape (H, W) dtype float32, valori in °C
        Oppure array vuoto (0, 0) se fallisce tutto.
    """
    # --- Strategia 1: SDK Python ---
    if _SDK_AVAILABLE:
        print(f"  [thermal] Strategia: SDK Python (libdirp.so) → {os.path.basename(rjpeg_path)}")
        return _extract_via_sdk(rjpeg_path, emissivity)

    # --- Strategia 2: CLI dji_irp ---
    if _DJI_IRP_CLI:
        print(f"  [thermal] Strategia: CLI dji_irp → {os.path.basename(rjpeg_path)}")
        return _extract_via_cli(rjpeg_path)

    # --- Strategia 3: Parsing raw ---
    print(f"  [thermal] Strategia: raw parsing (fallback) → {os.path.basename(rjpeg_path)}")
    return _extract_via_raw(rjpeg_path)


def _extract_via_sdk(rjpeg_path: str, emissivity: float) -> np.ndarray:
    """Usa dji_thermal_sdk + libdirp.so per estrarre la matrice temperature."""
    try:
        import ctypes
        from dji_thermal_sdk.dji_sdk import (
            dirp_create_from_rjpeg, dirp_get_rjpeg_resolution,
            dirp_measure_ex, dirp_destroy, DIRP_SUCCESS, dirp_resolution_t,
            dirp_measurement_params_t, dirp_set_measurement_params,
        )

        with open(rjpeg_path, "rb") as f:
            data = f.read()

        buf    = ctypes.create_string_buffer(data)
        size   = ctypes.c_int32(len(data))
        handle = ctypes.c_void_p()

        ret = dirp_create_from_rjpeg(buf, size, ctypes.byref(handle))
        if ret != DIRP_SUCCESS:
            return np.empty((0, 0), dtype=np.float32)

        # Imposta emissività
        try:
            mp = dirp_measurement_params_t()
            mp.emissivity        = ctypes.c_float(emissivity)
            mp.distance          = ctypes.c_float(5.0)
            mp.humidity          = ctypes.c_float(0.70)
            mp.reflected_temp    = ctypes.c_float(REFLECTED_TEMP_C)
            mp.atmospheric_temp  = ctypes.c_float(ATMOSPHERIC_TEMP_C)
            dirp_set_measurement_params(handle, ctypes.byref(mp))
        except Exception:
            pass

        res = dirp_resolution_t()
        dirp_get_rjpeg_resolution(handle, ctypes.byref(res))
        n   = res.width * res.height

        # Buffer float32 (4 byte/pixel) — confermato funzionante
        arr  = (ctypes.c_float * n)()
        ret2 = dirp_measure_ex(handle, arr, ctypes.c_int32(n * 4))
        dirp_destroy(handle)

        if ret2 != DIRP_SUCCESS:
            return np.empty((0, 0), dtype=np.float32)

        temps = np.frombuffer(arr, dtype=np.float32).reshape(res.height, res.width).copy()
        return temps

    except Exception as e:
        print(f"  [thermal_extractor] SDK error: {e}")
        return np.empty((0, 0), dtype=np.float32)


def _extract_via_cli(rjpeg_path: str) -> np.ndarray:
    """
    Usa dji_irp CLI per estrarre CSV temperature, poi li legge in numpy.
    dji_irp -s <input.jpg> -a process -o <output.csv>
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            out_csv = tmp.name

        result = subprocess.run(
            [_DJI_IRP_CLI, "-s", rjpeg_path, "-a", "process",
             "--measurefmt", "float32", "-o", out_csv],
            capture_output=True, timeout=30
        )

        if result.returncode == 0 and os.path.exists(out_csv):
            arr = np.loadtxt(out_csv, delimiter=",", dtype=np.float32)
            os.unlink(out_csv)
            return arr

        os.unlink(out_csv)
    except Exception as e:
        print(f"  [thermal_extractor] CLI error: {e}")

    return np.empty((0, 0), dtype=np.float32)


def _extract_via_raw(rjpeg_path: str) -> np.ndarray:
    """
    Parsing raw del RJPEG DJI: i dati termici sono un blob little-endian
    uint16 (in centesimi di Kelvin) dopo il marker APP4 (0xFFE4) o nei
    byte finali del file (DJI li appende dopo l'immagine JPEG visibile).

    Formula: T_C = (raw_uint16 / 100.0) - 273.15

    Nota: funziona per Zenmuse H20T/H30T, M30T, XT2.
    Per formati diversi potrebbe restituire array vuoto.
    """
    try:
        with open(rjpeg_path, "rb") as f:
            raw = f.read()

        # Cerca il magic DJI: "DJI\x00" o "DJIIR\x00" nei bytes finali
        # Il blob termico è tipicamente width×height uint16 little-endian
        # appeso dopo il marker EOI (0xFFD9) del JPEG normale

        eoi_pos = raw.rfind(b"\xff\xd9")
        if eoi_pos == -1:
            return np.empty((0, 0), dtype=np.float32)

        thermal_blob = raw[eoi_pos + 2:]

        # Dimensioni attese: 640×512 = 327680 bytes (uint16 = 655360 bytes)
        expected_uint16 = H30T_WIDTH_PX * H30T_HEIGHT_PX
        if len(thermal_blob) >= expected_uint16 * 2:
            arr_raw = np.frombuffer(
                thermal_blob[:expected_uint16 * 2],
                dtype=np.uint16
            ).reshape(H30T_HEIGHT_PX, H30T_WIDTH_PX)
            # Converti da centesimi di Kelvin → °C
            temp_c = arr_raw.astype(np.float32) / 100.0 - 273.15
            return temp_c

    except Exception as e:
        print(f"  [thermal_extractor] Raw parsing error: {e}")

    return np.empty((0, 0), dtype=np.float32)


# ==============================================================================
# ESTRAZIONE TEMPERATURA PANNELLO / HOTSPOT
# ==============================================================================

def extract_panel_temperature(
    rjpeg_path: str,
    mask_pixels: np.ndarray,
    mode: str = "mean",
    emissivity: float = EMISSIVITY_GLASS,
) -> float | None:
    """
    Estrae la temperatura rappresentativa di un pannello/hotspot
    dalla foto drone originale.

    Args:
        rjpeg_path  : path al file RJPEG DJI
        mask_pixels : array (N, 2) di coordinate (row, col) pixel nella foto
                      OPPURE maschera bool 2D (H, W)
        mode        : "mean" → temperatura media (pannello sano)
                      "max"  → temperatura massima (hotspot)
        emissivity  : default 0.95 (vetro)

    Returns:
        temperatura in °C, oppure None se estrazione fallisce
    """
    temp_matrix = get_temperature_matrix(rjpeg_path, emissivity)
    if temp_matrix.size == 0:
        return None

    H, W = temp_matrix.shape

    # Normalizza mask_pixels
    if isinstance(mask_pixels, np.ndarray) and mask_pixels.ndim == 2:
        if mask_pixels.shape == (H, W):
            # Maschera booleana 2D
            bool_mask = mask_pixels.astype(bool)
        else:
            # Array (N, 2) di coordinate (row, col)
            rows = mask_pixels[:, 0].clip(0, H - 1)
            cols = mask_pixels[:, 1].clip(0, W - 1)
            bool_mask = np.zeros((H, W), dtype=bool)
            bool_mask[rows, cols] = True
    else:
        return None

    values = temp_matrix[bool_mask]
    if values.size == 0:
        return None

    if mode == "max":
        return float(np.max(values))
    else:
        return float(np.mean(values))


def extract_from_patch_region(
    rjpeg_path: str,
    patch_offset_x: int,
    patch_offset_y: int,
    patch_mask: np.ndarray,
    mode: str = "mean",
    patch_size: int = 640,
    ortho_to_drone_scale: float = 1.0,
    emissivity: float = EMISSIVITY_GLASS,
) -> float | None:
    """
    Estrae temperatura da un pannello nella foto drone originale,
    sapendo l'offset della patch nell'ortomosaico e la maschera
    nella patch (coordinate locali 0..639).

    Il mapping è: coordinate_foto = (patch_offset + coord_patch) * ortho_to_drone_scale

    Args:
        rjpeg_path          : foto raw DJI
        patch_offset_x/y    : offset top-left della patch nell'ortomosaico (px)
        patch_mask          : maschera bool 2D (640, 640) del pannello nella patch
        mode                : "mean" o "max"
        patch_size          : 640
        ortho_to_drone_scale: rapporto GSD_ortomosaico / GSD_foto_drone
                              (di default 1.0 — stesso GSD)
        emissivity          : 0.95 (vetro)

    Returns:
        temperatura °C o None
    """
    temp_matrix = get_temperature_matrix(rjpeg_path, emissivity)
    if temp_matrix.size == 0:
        return None

    H_drone, W_drone = temp_matrix.shape

    # Converti coordinate patch → coordinate foto drone
    rows_patch, cols_patch = np.where(patch_mask)
    rows_drone = ((patch_offset_y + rows_patch) * ortho_to_drone_scale).astype(int)
    cols_drone = ((patch_offset_x + cols_patch) * ortho_to_drone_scale).astype(int)

    # Clip ai bordi
    valid = (rows_drone >= 0) & (rows_drone < H_drone) & \
            (cols_drone >= 0) & (cols_drone < W_drone)
    rows_drone = rows_drone[valid]
    cols_drone = cols_drone[valid]

    if rows_drone.size == 0:
        return None

    values = temp_matrix[rows_drone, cols_drone]
    if mode == "max":
        return float(np.max(values))
    return float(np.mean(values))


# ==============================================================================
# ESTRAZIONE TEMPERATURA VIA COORDINATE GPS (più accurata di pixel offset)
# ==============================================================================

def _gps_to_photo_pixel(
    lat_panel: float, lon_panel: float,
    lat_photo: float, lon_photo: float,
    altitude_m: float,
    hfov_deg: float, vfov_deg: float,
    img_w: int, img_h: int,
    heading_deg: float = 0.0,
) -> tuple[int, int] | None:
    """
    Converte coordinate GPS (lat_panel, lon_panel) in pixel (col, row)
    nella foto termica drone, usando la posizione GPS del centro foto,
    la quota AGL e il FOV del sensore.

    heading_deg: orientamento della camera (gimbal yaw, 0=Nord, 90=Est).
    Restituisce (col, row) o None se il pannello è fuori dal frame.
    """
    import math

    # Approssimazione planare (valida su distanze < 500 m)
    DEG_TO_M_LAT = 111_320.0
    DEG_TO_M_LON = 111_320.0 * math.cos(math.radians(lat_photo))

    dx_m = (lon_panel - lon_photo) * DEG_TO_M_LON   # Est positivo
    dy_m = (lat_panel - lat_photo) * DEG_TO_M_LAT   # Nord positivo

    # Ruota rispetto all'orientamento della camera
    heading_rad = math.radians(heading_deg)
    dx_cam =  dx_m * math.cos(heading_rad) + dy_m * math.sin(heading_rad)
    dy_cam = -dx_m * math.sin(heading_rad) + dy_m * math.cos(heading_rad)

    # Footprint del sensore a terra
    if altitude_m < 0.5:
        return None
    footprint_w = 2.0 * altitude_m * math.tan(math.radians(hfov_deg / 2.0))
    footprint_h = 2.0 * altitude_m * math.tan(math.radians(vfov_deg / 2.0))

    # Pixel: centro = (img_w/2, img_h/2)
    # dx_cam positivo → pixel a destra; dy_cam positivo → pixel in alto
    col = int(img_w / 2.0 + dx_cam / footprint_w * img_w)
    row = int(img_h / 2.0 - dy_cam / footprint_h * img_h)

    if 0 <= col < img_w and 0 <= row < img_h:
        return col, row
    return None


def extract_at_gps(
    rjpeg_path: str,
    lat_panel: float,
    lon_panel: float,
    mode: str = "max",
    radius_px: int = 8,
    emissivity: float = EMISSIVITY_GLASS,
) -> float | None:
    """
    Estrae la temperatura nella foto drone alla posizione GPS del pannello,
    usando il mapping GPS → pixel tramite EXIF (lat/lon/alt/FOV/heading).

    Questo è più accurato di extract_from_patch_region che usava pixel offset
    dell'ortomosaico (coordinate diverse dalla foto drone).

    Args:
        rjpeg_path : foto raw DJI RJPEG
        lat_panel  : latitudine del centroide del pannello
        lon_panel  : longitudine del centroide del pannello
        mode       : "max" per hotspot, "mean" per pannello sano
        radius_px  : raggio (pixel) del patch estratto attorno al centroide
        emissivity : 0.95 (vetro FV)

    Returns:
        temperatura °C, oppure None
    """
    # 1. Leggi GPS della foto drone (XMP DJI → fallback EXIF PIL DMS)
    lat_photo = 0.0
    lon_photo = 0.0
    alt_photo = 0.0
    heading   = 0.0

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
        import re as _re

        img_pil = Image.open(rjpeg_path)

        # Strategia A: XMP DJI (presente in alcuni RJPEG DJI)
        xmp = img_pil.info.get("xmp", b"").decode("utf-8", errors="ignore")
        def _xget(tag):
            m = _re.search(rf'drone-dji:{tag}="([^"]+)"', xmp)
            return m.group(1) if m else None

        _lat_x = _xget("GpsLatitude")
        _lon_x = _xget("GpsLongitude")
        if _lat_x and _lon_x:
            lat_photo = float(_lat_x)
            lon_photo = float(_lon_x)
            alt_photo = abs(float(_xget("RelativeAltitude") or "0"))
            heading   = float(_xget("GimbalYawDegree") or _xget("FlightYawDegree") or "0")

        # Strategia B: EXIF standard in formato DMS (PIL._getexif)
        if lat_photo == 0.0 and lon_photo == 0.0:
            exif = img_pil._getexif() or {}
            gps_raw = None
            for tag_id, val in exif.items():
                if TAGS.get(tag_id) == "GPSInfo":
                    gps_raw = {GPSTAGS.get(k, k): v for k, v in val.items()}
                    break
            if gps_raw:
                def _dms(dms, ref):
                    d, m, s = dms
                    dec = float(d) + float(m) / 60.0 + float(s) / 3600.0
                    return -dec if ref in ("S", "W") else dec

                lat_dms = gps_raw.get("GPSLatitude")
                lon_dms = gps_raw.get("GPSLongitude")
                if lat_dms and lon_dms:
                    lat_photo = _dms(lat_dms, gps_raw.get("GPSLatitudeRef",  "N"))
                    lon_photo = _dms(lon_dms, gps_raw.get("GPSLongitudeRef", "E"))
                    # GPSAltitude è ASL; usiamo il default AGL se troppo alto
                    alt_asl = float(gps_raw.get("GPSAltitude", 0.0))
                    alt_photo = alt_asl if alt_asl <= 80.0 else H30T_ALT_DEFAULT
                    heading = float(gps_raw.get("GPSTrack", 0.0))

    except Exception:
        pass

    if lat_photo == 0.0 and lon_photo == 0.0:
        return None

    # 2. Recupera risoluzione e FOV
    meta = get_exif_metadata(rjpeg_path)
    w    = meta["image_width"]
    h    = meta["image_height"]
    hfov = meta["fov_h_deg"]
    vfov = meta["fov_v_deg"]
    if alt_photo < 0.5:
        alt_photo = meta["altitude_m"]

    # 3. Converti GPS → pixel nella foto
    px = _gps_to_photo_pixel(lat_panel, lon_panel,
                              lat_photo, lon_photo,
                              alt_photo, hfov, vfov,
                              w, h, heading)
    if px is None:
        return None

    col, row = px

    # 4. Estrai matrice temperature e campiona il patch
    temp_matrix = get_temperature_matrix(rjpeg_path, emissivity)
    if temp_matrix.size == 0:
        return None

    H_t, W_t = temp_matrix.shape
    # Ricala col/row se la risoluzione termica differisce da quella EXIF
    col_t = int(col * W_t / w)
    row_t = int(row * H_t / h)

    r0 = max(0, row_t - radius_px)
    r1 = min(H_t, row_t + radius_px + 1)
    c0 = max(0, col_t - radius_px)
    c1 = min(W_t, col_t + radius_px + 1)

    patch = temp_matrix[r0:r1, c0:c1]
    if patch.size == 0:
        return None

    if mode == "max":
        return float(patch.max())
    return float(patch.mean())


# ==============================================================================
# UTILITY: PIXEL COUNT → AREA m²
# ==============================================================================

def pixels_to_area_m2(n_pixels: int, gsd_m_per_px: float) -> float:
    """
    Converte numero di pixel della maschera in area reale (m²).

    Args:
        n_pixels      : numero pixel nella maschera MaskDINO
        gsd_m_per_px  : GSD in m/pixel (dalla foto drone o da ortomosaico)

    Returns:
        area in m²
    """
    return n_pixels * (gsd_m_per_px ** 2)


# ==============================================================================
# MAIN — test standalone
# ==============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python thermal_extractor.py <foto.jpg>")
        sys.exit(0)

    path = sys.argv[1]
    print(f"\n{'='*55}")
    print(f"  File: {path}")
    print(f"{'='*55}")

    meta = get_exif_metadata(path)
    print(f"  Altitudine    : {meta['altitude_m']:.1f} m")
    print(f"  T. ambiente   : {meta['t_amb_c']:.1f} °C")
    print(f"  Emissività    : {meta['emissivity']}")
    print(f"  FOV (H×V)     : {meta['fov_h_deg']}° × {meta['fov_v_deg']}°")
    print(f"  Dimensioni    : {meta['image_width']}×{meta['image_height']} px")

    gsd = get_gsd(path)
    print(f"\n  GSD calcolato : {gsd*100:.2f} cm/pixel ({gsd:.5f} m/px)")

    print(f"\n  Estrazione matrice temperatura...")
    T = get_temperature_matrix(path)
    if T.size > 0:
        print(f"  Shape         : {T.shape}")
        print(f"  T min / max   : {T.min():.1f} °C / {T.max():.1f} °C")
        print(f"  T media       : {T.mean():.1f} °C")
    else:
        print("  ❌ Estrazione fallita (SDK/CLI/raw non disponibili)")

    print(f"{'='*55}\n")
