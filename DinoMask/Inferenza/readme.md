Guida installazione completa
Sistema (una volta sola)

apt-get install -y libexif12 exiftool
Python packages

/root/env/bin/pip install \
    simplekml \
    geopandas \
    pandas \
    numpy \
    opencv-python \
    Pillow \
    rasterio \
    pyproj \
    shapely \
    dji-thermal-sdk
File .so DJI (già fatto)
Questi file devono essere in ~/DinoMask/:


libdirp.so
libMicroIA_Release_x64.so
libMicroJPEG_Release_x64.so
libMicroTA_Release_x64.so
libv_cirp.so
libv_dirp.so
libv_girp.so
libv_hirp.so
libv_iirp.so
Scaricati da: dji_thermal_sdk_v1.8 → tsdk-core → lib → linux → release_x64

Struttura cartelle richiesta

~/DinoMask/
├── ortomosaico.tif          ← GeoTIFF georeferenziato
├── inferenza_patches/       ← generate con taglia_mosaico_training.py
└── foto_drone/              ← foto RJPEG originali DJI (.jpg)
Verifica tutto funzioni

cd ~/DinoMask
/root/env/bin/python -c "
import simplekml, geopandas, pandas, cv2, rasterio, pyproj, shapely
print('packages OK')
from thermal_extractor import get_temperature_matrix
print('DJI SDK OK')
from pvgis_client import get_pvgis_data
print('PVGIS OK')
from photo_matcher import load_drone_photos_index
print('photo_matcher OK')
"
Lancio pipeline

# 1. Taglia le patch
/root/env/bin/python taglia_mosaico_training.py --no-gui --input ortomosaico.tif --output inferenza_patches

# 2. Lancia inferenza
/root/env/bin/python inferenzaswin.py
