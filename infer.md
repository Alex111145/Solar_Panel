Riepilogo — inf.py
Uno script di inferenza e analisi per impianti fotovoltaici da immagini aeree (drone), basato su MaskDINO (segmentazione a istanza). Funziona in più fasi:

1. Rilevamento pannelli (MaskDINO)
Carica il modello addestrato (model_best.pth) con backbone Swin-L
Scorre le immagini di test (tile del mosaico drone)
Applica detection + segmentazione con soglia 0.50 e NMS post-processing
Classifica ogni pannello in 3 classi: PV_Module / Hotspot / Degrado
2. Geolocalizzazione
Legge il GeoTIFF del mosaico e il file .tfw per calcolare il GSD (cm/pixel)
Converte le coordinate pixel di ogni pannello in lat/lon WGS84 tramite rasterio + pyproj
Calcola l'area reale (m²) da numero di pixel × GSD²
Fonde rilevamenti duplicati entro 1 metro
3. Analisi termica (opzionale, DJI Thermal SDK)
Cerca i file termici accoppiati a ogni tile (RJPEG o TIFF radiometrico)
Corregge l'emissività tramite la legge di Stefan-Boltzmann
Estrae per ogni pannello: T_apparente, T_reale, efficienza termica η_reale
Hotspot → valore massimo; PV_Module/Degrado → media
4. Analisi irradiazione solare (PVGIS)
Chiede all'utente la città di installazione
Geocodifica con Nominatim → lat/lon
Chiama PVGIS API v5.2 (JRC Europa) per:
Irradiazione mensile orizzontale e su piano ottimale
Produzione kWh mensile/annuale stimata per l'impianto rilevato
CO₂ evitata (kg/anno)
5. Export
File	Contenuto
Mappa_Pannelli.kml/kmz	Punti geolocalizzati con score, classe, dati termici
Rilevamenti_Pannelli.csv	Tabella completa di ogni pannello
Rilevamenti_Pannelli.geojson	Poligoni o punti (per GIS)
Mosaico_Finale_Rilevato.jpg	Mosaico annotato con contorni colorati per classe
inference_results/	Patch singole annotate
