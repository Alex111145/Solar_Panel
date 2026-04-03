import math

# ==========================================
# 1. FUNZIONI TERMODINAMICHE (Capitolo 6)
# ==========================================

def calcola_t_reale(temp_apparente_c, emissivita, temp_amb_c):
    """Applica l'Eq. 3.1 di Stefan-Boltzmann per correggere l'emissività."""
    t_app_k = temp_apparente_c + 273.15
    t_amb_k = temp_amb_c + 273.15
    t_reale_k = ((t_app_k**4 - (1 - emissivita) * t_amb_k**4) / emissivita)**0.25
    return t_reale_k - 273.15

def calcola_efficienza(eta_nom, gamma, t_reale_c, t_stc=25.0):
    """Calcola l'efficienza termica declassata dal calore."""
    delta_t = t_reale_c - t_stc
    return eta_nom * (1 + gamma * delta_t)

# ==========================================
# 2. MOTORE DIAGNOSTICO
# ==========================================

def esegui_diagnostica_singola(
    t_sano_app, 
    t_hotspot_app, 
    emissivita, 
    t_amb, 
    tipo_pannello, 
    eta_nom, 
    gamma, 
    area, 
    irr, 
    esh, 
    giorni, 
    costo
):
    print("\n" + "="*70)
    print(" 🚀 REPORT DIAGNOSTICO IBRIDO UAV (O&M)")
    print("="*70)
    
    # 1. Conversione in Temperature Reali
    t_reale_sano = calcola_t_reale(t_sano_app, emissivita, t_amb)
    t_reale_hotspot = calcola_t_reale(t_hotspot_app, emissivita, t_amb)
    delta_t = t_reale_hotspot - t_reale_sano
    
    # 2. Calcolo Efficienze
    eta_sano = calcola_efficienza(eta_nom, gamma, t_reale_sano)
    eta_hotspot = calcola_efficienza(eta_nom, gamma, t_reale_hotspot)
    
    # 3. State of Health Relativo
    soh = (eta_hotspot / eta_sano) * 100.0
    
    # 4. Calcolo Potenze
    potenza_attesa = irr * area * eta_sano
    potenza_erogata = irr * area * eta_hotspot
    potenza_persa = potenza_attesa - potenza_erogata
    
    # 5. KPI Economici
    if potenza_persa > 0:
        energia_persa_kwh = (potenza_persa * esh) / 1000
        perdita_euro = energia_persa_kwh * giorni * costo
    else:
        energia_persa_kwh = 0
        perdita_euro = 0
        
    # 6. Matrice Decisionale (Tesi)
    if soh >= 90.0:
        if delta_t > 1.0:
            diagnosi = "SOILING / LIEVE ANOMALIA (Guasto Trascurabile)"
        else:
            diagnosi = "PANNELLO SANO (Nessuna anomalia termica rilevante)"
    else:
        diagnosi = "GUASTO FISICO GRAVE (Hotspot Severo)"
        
    # STAMPA REPORT
    print(f"Tecnologia Modulo  : {tipo_pannello} (Efficienza STC: {eta_nom*100:.2f}%)")
    print(f"Area Maschera (IA) : {area:.2f} m²")
    print(f"Irraggiamento (G)  : {irr:.2f} W/m²")
    print("-" * 70)
    print("1. PROFILO TERMICO (Baseline vs Hotspot)")
    print(f"Temp. Riflessa     : {t_amb:.2f} °C")
    print(f"Temp. Reale (Sano) : {t_reale_sano:.2f} °C (Baseline 100%)")
    print(f"Temp. Reale (Guasto): {t_reale_hotspot:.2f} °C (Picco IA)")
    print(f"Delta Termico (ΔT) : +{delta_t:.2f} °C")
    print("-" * 70)
    print("2. DIAGNOSI E STATO DI SALUTE (SoH)")
    print(f"Stato IA rilevato  : {diagnosi}")
    print(f"SoH Relativo       : {soh:.2f} % (100% = Pannello sano adiacente)")
    print("-" * 70)
    print("3. EFFICIENZA E POTENZA")
    print(f"Efficienza Attesa  : {eta_sano*100:.2f} % (al netto del meteo attuale)")
    print(f"Efficienza Hotspot : {eta_hotspot*100:.2f} %")
    print(f"Potenza Attesa     : {potenza_attesa:.2f} W")
    print(f"Potenza Erogata    : {potenza_erogata:.2f} W")
    if potenza_persa > 0:
        print(f"Potenza Dissipata  : {potenza_persa:.2f} W")
    print("-" * 70)
    print("4. IMPATTO ECONOMICO (O&M)")
    if perdita_euro > 0:
        print(f"Energia Persa/Giorno: {energia_persa_kwh:.3f} kWh")
        print(f"Mancato Guadagno    : -{perdita_euro:.2f} € / anno")
    else:
        print("Mancato Guadagno    : Nessuna perdita rilevante.")
    print("="*70 + "\n")

# ==========================================
# 3. INTERFACCIA E INSERIMENTO DATI
# ==========================================
if __name__ == "__main__":
    print("--- STRUMENTO DIAGNOSTICO PV ---")
    
    def get_input(testo, default=None):
        if default is not None:
            val = input(f"{testo} [Default: {default}]: ")
            return float(val) if val.strip() != "" else default
        else:
            val = input(f"{testo}: ")
            return float(val)
            
    # --- PROMEMORIA IMPORTANTE ---
    print("\n" + "*"*60)
    print(" PROMEMORIA CONVERSIONE TERMODINAMICA")
    print(" Inserisci le temperature così come le leggi a schermo dal")
    print(" software. Il programma si occuperà automaticamente di ")
    print(" convertirle in Temperature Reali compensando l'emissività")
    print(" e la temperatura riflessa, secondo l'Eq. 3.1 della tesi.")
    print("*"*60 + "\n")

    try:
        t_amb = get_input("1. Temp. Riflessa/Ambiente (°C)", 13.9)
        emissivita = get_input("2. Emissività del vetro (es. 0.95)", 0.95)
        
        print("\n--- DATI TERMICI DEL PANNELLO DA DIAGNOSTICARE ---")
        t_sano = get_input("3. Temp. Media dei pannelli SANI della riga (°C)")
        t_hotspot = get_input("4. Temp. MASSIMA dell'Hotspot rilevato (°C)")
        
        # --- MENU SELEZIONE PANNELLO ---
        print("\n--- TECNOLOGIA PANNELLO ---")
        print("   1) N-Type (Efficienza: 23.0%)")
        print("   2) Monocristallino (Efficienza: 20.0%)")
        print("   3) Policristallino (Efficienza: 16.5%)")
        scelta_tipo = input("5. Digita 1, 2 o 3 [Default: 3]: ")
        
        mappa_pannelli = {
            "1": {"tipo": "N-Type", "eta_nom": 0.23, "gamma": -0.0028},
            "2": {"tipo": "Monocristallino", "eta_nom": 0.20, "gamma": -0.0037},
            "3": {"tipo": "Policristallino", "eta_nom": 0.165, "gamma": -0.0042}
        }
        
        dati_pannello = mappa_pannelli.get(scelta_tipo.strip(), mappa_pannelli["3"])
        
        print("\n--- DATI GEOMETRICI E AMBIENTALI ---")
        area = get_input("6. Area del modulo (m²)", 1.63)
        irr = get_input("7. Irraggiamento attuale (W/m²)", 900.0)
        esh = get_input("8. Ore Equivalenti (ESH)", 3.18)
        giorni = get_input("9. Giorni Utili annui", 300)
        costo = get_input("10. Costo energia (€/kWh)", 0.40)
        
        # Avvio elaborazione
        esegui_diagnostica_singola(
            t_sano_app=t_sano, 
            t_hotspot_app=t_hotspot, 
            emissivita=emissivita, 
            t_amb=t_amb, 
            tipo_pannello=dati_pannello['tipo'], 
            eta_nom=dati_pannello['eta_nom'], 
            gamma=dati_pannello['gamma'], 
            area=area, 
            irr=irr, 
            esh=esh, 
            giorni=giorni, 
            costo=costo
        )

    except ValueError:
        print("\n[ERRORE] Formato non valido. Usa il punto per i decimali (es. 13.8) e inserisci solo numeri.")
