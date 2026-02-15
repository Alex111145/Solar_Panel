import simplekml
import os

def check_anchor():
    print("\n" + "="*50)
    print("🎯 GENERAZIONE PUNTO DI PRECISIONE 2D")
    print("="*50)
    
    try:
        lat = float(input("\n📍 Latitudine (es. 45.840736): ").strip().replace(',', '.'))
        lon = float(input("📍 Longitudine (es. 8.790503): ").strip().replace(',', '.'))
        
        kml = simplekml.Kml()
        pnt = kml.newpoint(name="Punto_Ancora_Esatto", coords=[(lon, lat)])
        
        # CAMBIO ICONA: Usiamo un cerchietto piccolo invece della puntina
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
        pnt.style.iconstyle.color = 'ff0000ff' # Rosso pieno
        pnt.style.iconstyle.scale = 0.5        # Lo rendiamo piccolo per non coprire lo spigolo
        
        output_file = "punto_ancora_2d.kml"
        kml.save(output_file)
        
        print("-" * 50)
        print(f"✅ PUNTO CREATO: {output_file}")
        print("👉 Trascinalo su Google Earth.")
        print("-" * 50)
        
    except ValueError:
        print("❌ Inserisci numeri validi.")

if __name__ == "__main__":
    check_anchor()