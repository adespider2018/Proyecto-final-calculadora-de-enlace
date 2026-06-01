# -*- coding: utf-8 -*-
"""
Calculadora de enlace satelital SATMEX 5 — versión mejorada
Materia: Sistemas de Comunicaciones
Integrantes: Jesús Ademar Santillán Domínguez y Carlos Hernández Pacheco

Ejecución:
    streamlit run app_enlace_satelital_satmex5_mejorado.py

Notas:
- El programa está orientado al cálculo académico de presupuesto de enlace satelital.
- Los datos SATMEX 5 de frecuencias, polarizaciones, regiones, ATP, back-off y C/I se cargan de forma interna.
- Los valores de modulación, FEC, Eb/No y roll-off son editables; los predeterminados son referencias típicas para cálculos académicos.
- Los valores por estación terrena incluyen datos de ejemplo basados en tablas de SATMEX 5 para México. Si se requiere una localidad no incluida,
  usar la opción de estación manual y capturar EIRP/G/T/SFD manualmente.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ================================================================
# Constantes físicas y orbitales
# ================================================================
C_LIGHT = 300_000_000.0  # m/s; se usa 3x10^8 para coincidir con el procedimiento del proyecto
K_BOLTZMANN_DB = 228.6  # equivale a restar k = -228.6 dBW/K/Hz
EARTH_RADIUS_KM = 6378.137  # radio ecuatorial re [km]
EARTH_POLAR_RADIUS_KM = 6356.752  # radio polar b [km]
GEO_RADIUS_KM = 42164.170  # radio orbital geoestacionario rs [km]
SATMEX5_LON_W_DEG = 116.8
DEFAULT_TRANSPONDER_BW_MHZ = 36.0


# ================================================================
# Tablas base
# ================================================================
def create_frequency_plan() -> pd.DataFrame:
    """Genera el plan de frecuencias de SATMEX 5 para C y Ku."""
    rows = []

    # Banda C: odd H/V, even V/H, región C1.
    odd_c = list(range(1, 24, 2))
    even_c = list(range(2, 25, 2))
    for idx, tp in enumerate(odd_c):
        rows.append({
            "tp": f"{tp}N",
            "band": "C",
            "uplink_GHz": round(5.945 + 0.040 * idx, 3),
            "downlink_GHz": round(3.720 + 0.040 * idx, 3),
            "polarization": "H/V",
            "region": "C1",
            "bw_MHz": DEFAULT_TRANSPONDER_BW_MHZ,
        })
    for idx, tp in enumerate(even_c):
        rows.append({
            "tp": f"{tp}N",
            "band": "C",
            "uplink_GHz": round(5.965 + 0.040 * idx, 3),
            "downlink_GHz": round(3.740 + 0.040 * idx, 3),
            "polarization": "V/H",
            "region": "C1",
            "bw_MHz": DEFAULT_TRANSPONDER_BW_MHZ,
        })

    # Banda Ku: odd V/H; even H/V.
    odd_ku = list(range(1, 24, 2))
    even_ku = list(range(2, 25, 2))

    def ku_region(tp_num: int) -> str:
        if tp_num in [1, 3, 5, 7, 2, 4, 6, 8]:
            return "Ku1"
        if tp_num in [9, 11, 13, 15, 10, 12, 14, 16]:
            return "Ku1/2"
        return "Ku2"

    for idx, tp in enumerate(odd_ku):
        rows.append({
            "tp": f"{tp}K",
            "band": "Ku",
            "uplink_GHz": round(14.020 + 0.040 * idx, 3),
            "downlink_GHz": round(11.720 + 0.040 * idx, 3),
            "polarization": "V/H",
            "region": ku_region(tp),
            "bw_MHz": DEFAULT_TRANSPONDER_BW_MHZ,
        })
    for idx, tp in enumerate(even_ku):
        rows.append({
            "tp": f"{tp}K",
            "band": "Ku",
            "uplink_GHz": round(14.040 + 0.040 * idx, 3),
            "downlink_GHz": round(11.740 + 0.040 * idx, 3),
            "polarization": "H/V",
            "region": ku_region(tp),
            "bw_MHz": DEFAULT_TRANSPONDER_BW_MHZ,
        })

    return pd.DataFrame(rows)


SAT_PARAMS = {
    "C": {
        "ATP_dB": 5.0,
        "IBO_single_dB": 1.0,
        "OBO_single_dB": 0.3,
        "IBO_multi_dB": 8.0,
        "OBO_multi_dB": 5.0,
        "C_I_intermod_up_dB": 30.0,
        "C_I_intermod_down_dB": 16.0,
        "C_I_xpol_up_dB": 28.0,
        "C_I_xpol_down_dB": 32.0,
        "C_I_adj_up_dB": 34.0,
        "C_I_adj_down_dB": 30.0,
    },
    "Ku": {
        "ATP_dB": 14.0,
        "IBO_single_dB": 0.0,
        "OBO_single_dB": 0.0,
        "IBO_multi_dB": 8.0,
        "OBO_multi_dB": 5.0,
        "C_I_intermod_up_dB": 35.0,
        "C_I_intermod_down_dB": 18.0,
        "C_I_xpol_up_dB": 29.0,
        "C_I_xpol_down_dB": 30.0,
        "C_I_adj_up_dB": 39.0,
        "C_I_adj_down_dB": 28.0,
    },
}

MODULATION_PRESETS = {
    "2-FSK": {
        "family": "FSK",
        "bits_per_symbol": 1,
        "fec": "1/2",
        "rolloff": 0.35,
        "ebno_dB": 11.0,
        "note": "FSK es robusta pero menos eficiente espectralmente; úsala si se requiere simplicidad/robustez.",
    },
    "BPSK": {
        "family": "PSK",
        "bits_per_symbol": 1,
        "fec": "1/2",
        "rolloff": 0.35,
        "ebno_dB": 5.0,
        "note": "Muy robusta; ancho de banda mayor para la misma tasa.",
    },
    "QPSK / 4PSK": {
        "family": "PSK",
        "bits_per_symbol": 2,
        "fec": "1/2",
        "rolloff": 0.30,
        "ebno_dB": 7.2,
        "note": "Opción equilibrada para un enlace académico. Valores predeterminados: roll-off 0.30, FEC 1/2 y Eb/No 7.2 dB.",
    },
    "8PSK": {
        "family": "PSK",
        "bits_per_symbol": 3,
        "fec": "2/3",
        "rolloff": 0.25,
        "ebno_dB": 9.0,
        "note": "Mayor eficiencia espectral, pero requiere mejor C/N y más linealidad.",
    },
    "16QAM": {
        "family": "QAM",
        "bits_per_symbol": 4,
        "fec": "3/4",
        "rolloff": 0.25,
        "ebno_dB": 12.0,
        "note": "Eficiente en ancho de banda, sensible a ruido/no linealidad.",
    },
    "64QAM": {
        "family": "QAM",
        "bits_per_symbol": 6,
        "fec": "5/6",
        "rolloff": 0.20,
        "ebno_dB": 16.0,
        "note": "Alta eficiencia espectral; no recomendada si el margen del enlace es bajo.",
    },
}

# Estaciones de México. Longitud en grados Oeste. Altitudes aproximadas y editables.
# Los valores de EIRP/G/T/SFD por estación son valores de trabajo para la calculadora, tomados/ordenados de las tablas SATMEX 5.
STATION_DATA = {
    "ACAPULCO, GRO.": {
        "lat": 16.85, "lon_w": 99.92, "alt_m": 30,
        "C": {"V/H": {"eirp": 40.32, "gt": -0.76, "sfd": -94.77}, "H/V": {"eirp": 40.70, "gt": -0.43, "sfd": -94.30}},
        "Ku1": {"V/H": {"eirp": 48.31, "gt": 0.09, "sfd": -93.25}, "H/V": {"eirp": 49.24, "gt": -1.22, "sfd": -91.38}},
        "Ku2": {"V/H": {"eirp": 47.62, "gt": 2.34, "sfd": -98.00}, "H/V": {"eirp": 48.87, "gt": 1.70, "sfd": -98.29}},
    },
    "CANCUN, Q.ROO": {
        "lat": 21.08, "lon_w": 86.77, "alt_m": 10,
        "C": {"V/H": {"eirp": 41.29, "gt": 3.16, "sfd": -98.01}, "H/V": {"eirp": 40.94, "gt": 2.81, "sfd": -98.22}},
        "Ku1": {"V/H": {"eirp": 51.38, "gt": 3.55, "sfd": -96.71}, "H/V": {"eirp": 51.19, "gt": 3.63, "sfd": -96.23}},
        "Ku2": {"V/H": {"eirp": 46.58, "gt": 1.15, "sfd": -96.81}, "H/V": {"eirp": 47.77, "gt": -0.13, "sfd": -96.46}},
    },
    "CIUDAD JUAREZ, CHIH.": {
        "lat": 31.73, "lon_w": 106.48, "alt_m": 1137,
        "C": {"V/H": {"eirp": 40.95, "gt": -1.09, "sfd": -94.92}, "H/V": {"eirp": 40.95, "gt": -0.28, "sfd": -93.97}},
        "Ku1": {"V/H": {"eirp": 51.32, "gt": 3.81, "sfd": -96.97}, "H/V": {"eirp": 51.31, "gt": 3.76, "sfd": -96.36}},
        "Ku2": {"V/H": {"eirp": 48.06, "gt": 0.82, "sfd": -96.48}, "H/V": {"eirp": 48.48, "gt": 0.44, "sfd": -97.03}},
    },
    "CULIACAN, SIN.": {
        "lat": 24.80, "lon_w": 107.40, "alt_m": 60,
        "C": {"V/H": {"eirp": 41.17, "gt": 0.45, "sfd": -96.22}, "H/V": {"eirp": 41.17, "gt": 1.02, "sfd": -95.51}},
        "Ku1": {"V/H": {"eirp": 51.10, "gt": 4.79, "sfd": -97.95}, "H/V": {"eirp": 51.28, "gt": 5.25, "sfd": -97.85}},
        "Ku2": {"V/H": {"eirp": 47.17, "gt": 1.47, "sfd": -97.13}, "H/V": {"eirp": 48.01, "gt": 1.23, "sfd": -97.82}},
    },
    "GUADALAJARA, JAL.": {
        "lat": 20.67, "lon_w": 103.33, "alt_m": 1566,
        "C": {"V/H": {"eirp": 40.74, "gt": -0.54, "sfd": -95.57}, "H/V": {"eirp": 40.89, "gt": 0.37, "sfd": -94.52}},
        "Ku1": {"V/H": {"eirp": 50.85, "gt": 3.84, "sfd": -97.00}, "H/V": {"eirp": 51.32, "gt": 3.26, "sfd": -95.86}},
        "Ku2": {"V/H": {"eirp": 48.01, "gt": 3.45, "sfd": -99.11}, "H/V": {"eirp": 48.58, "gt": 2.59, "sfd": -99.18}},
    },
    "HERMOSILLO, SON.": {
        "lat": 29.07, "lon_w": 110.97, "alt_m": 210,
        "C": {"V/H": {"eirp": 40.93, "gt": -0.85, "sfd": -95.16}, "H/V": {"eirp": 40.51, "gt": -0.04, "sfd": -94.21}},
        "Ku1": {"V/H": {"eirp": 51.36, "gt": 4.50, "sfd": -97.66}, "H/V": {"eirp": 50.40, "gt": 5.12, "sfd": -97.72}},
        "Ku2": {"V/H": {"eirp": 47.39, "gt": 0.43, "sfd": -96.09}, "H/V": {"eirp": 48.19, "gt": 0.31, "sfd": -96.90}},
    },
    "LEON, GTO.": {
        "lat": 21.17, "lon_w": 101.70, "alt_m": 1815,
        "C": {"V/H": {"eirp": 40.88, "gt": 0.01, "sfd": -95.96}, "H/V": {"eirp": 41.16, "gt": 0.76, "sfd": -95.07}},
        "Ku1": {"V/H": {"eirp": 51.35, "gt": 5.34, "sfd": -98.50}, "H/V": {"eirp": 51.79, "gt": 4.50, "sfd": -97.10}},
        "Ku2": {"V/H": {"eirp": 47.96, "gt": 3.57, "sfd": -99.23}, "H/V": {"eirp": 48.60, "gt": 2.17, "sfd": -98.76}},
    },
    "MATAMOROS, TAMPS.": {
        "lat": 25.88, "lon_w": 97.50, "alt_m": 10,
        "C": {"V/H": {"eirp": 40.90, "gt": 0.20, "sfd": -95.80}, "H/V": {"eirp": 41.00, "gt": 0.50, "sfd": -95.40}},
        "Ku1": {"V/H": {"eirp": 50.70, "gt": 5.06, "sfd": -98.22}, "H/V": {"eirp": 52.09, "gt": 2.98, "sfd": -95.58}},
        "Ku2": {"V/H": {"eirp": 47.71, "gt": 2.86, "sfd": -98.53}, "H/V": {"eirp": 48.65, "gt": 2.01, "sfd": -98.60}},
    },
    "MAZATLAN, SIN.": {
        "lat": 23.22, "lon_w": 106.42, "alt_m": 10,
        "C": {"V/H": {"eirp": 40.80, "gt": -0.42, "sfd": -95.75}, "H/V": {"eirp": 40.72, "gt": 0.55, "sfd": -94.64}},
        "Ku1": {"V/H": {"eirp": 51.28, "gt": 3.92, "sfd": -97.08}, "H/V": {"eirp": 50.88, "gt": 3.99, "sfd": -96.59}},
        "Ku2": {"V/H": {"eirp": 47.60, "gt": 2.17, "sfd": -97.83}, "H/V": {"eirp": 48.23, "gt": 1.83, "sfd": -98.42}},
    },
    "MERIDA, YUC.": {
        "lat": 20.97, "lon_w": 89.62, "alt_m": 10,
        "C": {"V/H": {"eirp": 41.12, "gt": 3.01, "sfd": -98.01}, "H/V": {"eirp": 41.64, "gt": 2.81, "sfd": -98.07}},
        "Ku1": {"V/H": {"eirp": 51.76, "gt": 5.42, "sfd": -98.58}, "H/V": {"eirp": 51.80, "gt": 5.78, "sfd": -98.38}},
        "Ku2": {"V/H": {"eirp": 46.73, "gt": 1.87, "sfd": -97.53}, "H/V": {"eirp": 47.74, "gt": 1.11, "sfd": -97.70}},
    },
    "MEXICALI, BCN.": {
        "lat": 32.67, "lon_w": 115.48, "alt_m": 8,
        "C": {"V/H": {"eirp": 40.74, "gt": -1.75, "sfd": -94.68}, "H/V": {"eirp": 40.28, "gt": -0.52, "sfd": -93.31}},
        "Ku1": {"V/H": {"eirp": 51.79, "gt": 3.52, "sfd": -96.68}, "H/V": {"eirp": 50.51, "gt": 4.62, "sfd": -97.22}},
        "Ku2": {"V/H": {"eirp": 49.15, "gt": 0.81, "sfd": -96.47}, "H/V": {"eirp": 47.94, "gt": 1.67, "sfd": -98.26}},
    },
    "MEXICO, D.F.": {
        "lat": 19.40, "lon_w": 99.15, "alt_m": 2233,
        "C": {"V/H": {"eirp": 40.78, "gt": 0.37, "sfd": -96.04}, "H/V": {"eirp": 41.18, "gt": 0.84, "sfd": -95.43}},
        "Ku1": {"V/H": {"eirp": 51.01, "gt": 4.10, "sfd": -97.26}, "H/V": {"eirp": 51.29, "gt": 2.94, "sfd": -95.54}},
        "Ku2": {"V/H": {"eirp": 47.69, "gt": 3.60, "sfd": -99.26}, "H/V": {"eirp": 49.04, "gt": 2.43, "sfd": -99.02}},
    },
    "MONTERREY, N.L.": {
        "lat": 25.67, "lon_w": 100.32, "alt_m": 540,
        "C": {"V/H": {"eirp": 41.22, "gt": 0.57, "sfd": -96.22}, "H/V": {"eirp": 41.47, "gt": 1.02, "sfd": -95.63}},
        "Ku1": {"V/H": {"eirp": 51.15, "gt": 5.44, "sfd": -98.60}, "H/V": {"eirp": 52.16, "gt": 3.87, "sfd": -96.47}},
        "Ku2": {"V/H": {"eirp": 47.59, "gt": 2.78, "sfd": -98.44}, "H/V": {"eirp": 48.69, "gt": 1.67, "sfd": -98.26}},
    },
    "MORELIA, MICH.": {
        "lat": 19.70, "lon_w": 101.12, "alt_m": 1920,
        "C": {"V/H": {"eirp": 40.75, "gt": -0.22, "sfd": -95.68}, "H/V": {"eirp": 41.04, "gt": 0.48, "sfd": -94.84}},
        "Ku1": {"V/H": {"eirp": 50.49, "gt": 3.80, "sfd": -96.96}, "H/V": {"eirp": 51.19, "gt": 2.35, "sfd": -94.95}},
        "Ku2": {"V/H": {"eirp": 47.91, "gt": 3.56, "sfd": -99.22}, "H/V": {"eirp": 48.74, "gt": 2.27, "sfd": -98.86}},
    },
    "OAXACA, OAX.": {
        "lat": 17.05, "lon_w": 96.72, "alt_m": 1555,
        "C": {"V/H": {"eirp": 40.51, "gt": 0.58, "sfd": -95.94}, "H/V": {"eirp": 40.60, "gt": 0.74, "sfd": -95.64}},
        "Ku1": {"V/H": {"eirp": 49.67, "gt": 3.40, "sfd": -96.56}, "H/V": {"eirp": 51.66, "gt": 1.33, "sfd": -93.93}},
        "Ku2": {"V/H": {"eirp": 47.68, "gt": 3.06, "sfd": -98.72}, "H/V": {"eirp": 48.95, "gt": 1.59, "sfd": -98.18}},
    },
    "PUEBLA, PUE.": {
        "lat": 19.05, "lon_w": 98.20, "alt_m": 2135,
        "C": {"V/H": {"eirp": 40.57, "gt": -1.02, "sfd": -95.28}, "H/V": {"eirp": 41.52, "gt": 0.08, "sfd": -94.04}},
        "Ku1": {"V/H": {"eirp": 51.06, "gt": 4.62, "sfd": -97.78}, "H/V": {"eirp": 51.60, "gt": 3.21, "sfd": -95.81}},
        "Ku2": {"V/H": {"eirp": 47.66, "gt": 3.76, "sfd": -99.42}, "H/V": {"eirp": 49.10, "gt": 2.57, "sfd": -99.16}},
    },
    "TAMPICO, TAMPS.": {
        "lat": 22.22, "lon_w": 97.85, "alt_m": 10,
        "C": {"V/H": {"eirp": 41.05, "gt": 1.20, "sfd": -96.72}, "H/V": {"eirp": 41.36, "gt": 1.52, "sfd": -96.26}},
        "Ku1": {"V/H": {"eirp": 51.46, "gt": 5.23, "sfd": -98.39}, "H/V": {"eirp": 52.09, "gt": 4.94, "sfd": -97.54}},
        "Ku2": {"V/H": {"eirp": 48.17, "gt": 3.25, "sfd": -98.91}, "H/V": {"eirp": 48.72, "gt": 1.91, "sfd": -98.50}},
    },
    "TIJUANA, BCN.": {
        "lat": 32.37, "lon_w": 117.02, "alt_m": 120,
        "C": {"V/H": {"eirp": 40.66, "gt": -1.92, "sfd": -94.53}, "H/V": {"eirp": 40.11, "gt": -0.67, "sfd": -93.14}},
        "Ku1": {"V/H": {"eirp": 51.25, "gt": 1.68, "sfd": -94.84}, "H/V": {"eirp": 49.74, "gt": 3.40, "sfd": -96.00}},
        "Ku2": {"V/H": {"eirp": 49.13, "gt": 1.08, "sfd": -96.74}, "H/V": {"eirp": 48.27, "gt": 1.69, "sfd": -98.28}},
    },
    "VERACRUZ, VER.": {
        "lat": 19.20, "lon_w": 96.13, "alt_m": 10,
        "C": {"V/H": {"eirp": 40.81, "gt": 1.35, "sfd": -96.76}, "H/V": {"eirp": 41.36, "gt": 1.56, "sfd": -96.41}},
        "Ku1": {"V/H": {"eirp": 51.53, "gt": 6.04, "sfd": -99.20}, "H/V": {"eirp": 52.44, "gt": 4.36, "sfd": -96.96}},
        "Ku2": {"V/H": {"eirp": 47.95, "gt": 4.22, "sfd": -99.88}, "H/V": {"eirp": 48.97, "gt": 2.51, "sfd": -99.10}},
    },
    "VILLAHERMOSA, TAB.": {
        "lat": 17.98, "lon_w": 92.92, "alt_m": 10,
        "C": {"V/H": {"eirp": 40.74, "gt": 2.13, "sfd": -97.39}, "H/V": {"eirp": 41.36, "gt": 2.19, "sfd": -97.19}},
        "Ku1": {"V/H": {"eirp": 51.72, "gt": 6.21, "sfd": -99.37}, "H/V": {"eirp": 52.35, "gt": 4.66, "sfd": -97.26}},
        "Ku2": {"V/H": {"eirp": 49.32, "gt": 3.94, "sfd": -99.60}, "H/V": {"eirp": 49.27, "gt": 2.87, "sfd": -99.46}},
    },
}

DEFAULT_SERVICES = pd.DataFrame([
    {"Servicio": "Canales de voz", "Cantidad": 5, "kbps_por_canal": 16.0},
    {"Servicio": "Canales de datos", "Cantidad": 10, "kbps_por_canal": 32.0},
    {"Servicio": "LAN", "Cantidad": 10, "kbps_por_canal": 512.0},
    {"Servicio": "Video", "Cantidad": 2, "kbps_por_canal": 384.0},
])


# ================================================================
# Utilidades de cálculo
# ================================================================
def db_to_linear(x_db: float) -> float:
    return 10 ** (x_db / 10.0)


def linear_to_db(x: float) -> float:
    if x <= 0:
        return float("-inf")
    return 10.0 * math.log10(x)


def inv_sum_db(values_db: List[float]) -> float:
    """Combina razones en dB por suma inversa: 1/Xtot = sum(1/Xi)."""
    inv = sum(1.0 / db_to_linear(v) for v in values_db if np.isfinite(v))
    if inv <= 0:
        return float("inf")
    return linear_to_db(1.0 / inv)


def parse_fec(fec_text: str) -> float:
    fec_text = str(fec_text).strip()
    if "/" in fec_text:
        a, b = fec_text.split("/", 1)
        return float(a) / float(b)
    return float(fec_text)


def antenna_gain_dbi(diameter_m: float, efficiency: float, freq_GHz: float) -> float:
    wavelength_m = C_LIGHT / (freq_GHz * 1e9)
    gain_lin = efficiency * (math.pi * diameter_m / wavelength_m) ** 2
    return linear_to_db(gain_lin)


def free_space_loss_db(distance_km: float, freq_GHz: float) -> float:
    """Pérdida por espacio libre: LFS = 20log10(4πd/λ)."""
    wavelength_m = C_LIGHT / (freq_GHz * 1e9)
    distance_m = distance_km * 1000.0
    return 20.0 * math.log10(4.0 * math.pi * distance_m / wavelength_m)


def geo_procedure(lat_deg: float, lon_w_deg: float, alt_m: float, sat_lon_w_deg: float = SATMEX5_LON_W_DEG) -> Dict[str, float]:
    """Geometría exactamente como la guía: B, gamma, rt, d, elevación y azimut."""
    H_km = alt_m / 1000.0
    B_deg = abs(sat_lon_w_deg - lon_w_deg)
    lat_rad = math.radians(lat_deg)
    B_rad = math.radians(B_deg)
    gamma_rad = math.acos(math.cos(lat_rad) * math.cos(B_rad))

    re = EARTH_RADIUS_KM
    b = EARTH_POLAR_RADIUS_KM
    numerator = (re**2 * math.cos(lat_rad))**2 + (b**2 * math.sin(lat_rad))**2
    denominator = (re * math.cos(lat_rad))**2 + (b * math.sin(lat_rad))**2
    rt_km = math.sqrt(numerator / denominator) + H_km

    rs = GEO_RADIUS_KM
    distance_km = math.sqrt(rt_km**2 + rs**2 - 2.0 * rt_km * rs * math.cos(gamma_rad))
    el_deg = math.degrees(math.acos(rs * math.sin(gamma_rad) / distance_km))
    az_deg = 180.0 + math.degrees(math.atan(math.tan(B_rad) / math.sin(lat_rad)))

    return {
        "B_deg": B_deg,
        "gamma_deg": math.degrees(gamma_rad),
        "rt_km": rt_km,
        "distance_km": distance_km,
        "az_deg": az_deg,
        "el_deg": el_deg,
    }


def look_angles(lat_deg: float, lon_w_deg: float, alt_m: float, sat_lon_w_deg: float = SATMEX5_LON_W_DEG) -> Dict[str, float]:
    return geo_procedure(lat_deg, lon_w_deg, alt_m, sat_lon_w_deg)


def get_region_key(band: str, region: str) -> str:
    if band == "C":
        return "C"
    if region == "Ku2":
        return "Ku2"
    # Para Ku1/2, se usa Ku1 como punto de partida para México; puede editarse manualmente.
    return "Ku1"


def get_sat_footprint(station_name: str, band: str, region: str, polarization: str) -> Dict[str, float]:
    station = STATION_DATA[station_name]
    region_key = get_region_key(band, region)
    if region_key not in station:
        region_key = "Ku1" if band == "Ku" else "C"
    pol = polarization if polarization in station[region_key] else list(station[region_key].keys())[0]
    return station[region_key][pol].copy()


def occupied_bandwidth_hz(rb_kbps: float, bits_per_symbol: int, fec_rate: float, rolloff: float) -> Tuple[float, float]:
    rb_bps = rb_kbps * 1000.0
    symbol_rate = rb_bps / (bits_per_symbol * fec_rate)
    bandwidth_hz = symbol_rate * (1.0 + rolloff)
    return symbol_rate, bandwidth_hz


def cn0_uplink_dbhz(eirp_tx_dBW: float, distance_km: float, freq_GHz: float, gt_sat_dB_K: float, losses_dB: float) -> float:
    lfs = free_space_loss_db(distance_km, freq_GHz)
    return eirp_tx_dBW - lfs - losses_dB + gt_sat_dB_K + K_BOLTZMANN_DB


def cn0_downlink_dbhz(eirp_sat_dBW: float, distance_km: float, freq_GHz: float, gt_rx_dB_K: float, losses_dB: float) -> float:
    lfs = free_space_loss_db(distance_km, freq_GHz)
    return eirp_sat_dBW - lfs - losses_dB + gt_rx_dB_K + K_BOLTZMANN_DB


def create_diagram(results: Dict[str, float | str]) -> go.Figure:
    """Diagrama visual del enlace con valores principales y conclusión."""
    valid = bool(results["valid"])
    color_main = "#163B73"
    color_up = "#1E88E5"
    color_down = "#7B1FA2"
    color_ok = "#0B7A3B"
    color_bad = "#B42318"
    status_color = color_ok if valid else color_bad
    status_fill = "#EAF7EF" if valid else "#FDECEC"

    fig = go.Figure()
    fig.update_layout(
        height=610,
        margin=dict(l=15, r=15, t=55, b=15),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(visible=False, range=[0, 12]),
        yaxis=dict(visible=False, range=[0, 7]),
        title=dict(
            text="<b>Esquema final del enlace SATMEX 5</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=23, color=color_main),
        ),
        showlegend=False,
    )

    # Tarjetas principales
    cards = [
        (0.35, 1.45, 3.25, 5.35, "ET transmisora", results["tx_name"], "#EEF6FF"),
        (4.45, 3.85, 7.55, 6.35, "SATMEX 5", "116.8° W", "#FFF7E6"),
        (8.75, 1.45, 11.65, 5.35, "ET receptora", results["rx_name"], "#F7F0FF"),
    ]
    for x0, y0, x1, y1, title, subtitle, fill in cards:
        fig.add_shape(
            type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(width=2.5, color=color_main), fillcolor=fill,
            layer="below",
        )
        fig.add_annotation(
            x=(x0+x1)/2, y=y1-0.45,
            text=f"<b>{title}</b><br><span style='font-size:13px'>{subtitle}</span>",
            showarrow=False, align="center", font=dict(size=16, color=color_main),
        )

    # Flechas
    fig.add_annotation(x=4.45, y=5.08, ax=3.25, ay=4.65, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowsize=1.25, arrowwidth=4, arrowcolor=color_up)
    fig.add_annotation(x=8.75, y=4.65, ax=7.55, ay=5.08, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowsize=1.25, arrowwidth=4, arrowcolor=color_down)

    uplink_text = (
        f"<b>Enlace ascendente</b><br>"
        f"f↑ = {results['uplink_GHz']:.3f} GHz<br>"
        f"d↑ = {results['d_up_km']:.2f} km<br>"
        f"LFS↑ = {results['lfs_up_dB']:.2f} dB<br>"
        f"C/N↑ = {results['cn_up_dB']:.2f} dB"
    )
    downlink_text = (
        f"<b>Enlace descendente</b><br>"
        f"f↓ = {results['downlink_GHz']:.3f} GHz<br>"
        f"d↓ = {results['d_down_km']:.2f} km<br>"
        f"LFS↓ = {results['lfs_down_dB']:.2f} dB<br>"
        f"C/N↓ = {results['cn_down_dB']:.2f} dB"
    )
    fig.add_annotation(x=3.75, y=6.15, text=uplink_text, showarrow=False, align="center",
                       bgcolor="#FFFFFF", bordercolor=color_up, borderwidth=2, font=dict(size=13))
    fig.add_annotation(x=8.25, y=6.15, text=downlink_text, showarrow=False, align="center",
                       bgcolor="#FFFFFF", bordercolor=color_down, borderwidth=2, font=dict(size=13))

    tx_text = (
        f"Az = {results['az_tx_deg']:.2f}°<br>El = {results['el_tx_deg']:.2f}°<br>"
        f"Gtx = {results['g_tx_dBi']:.2f} dBi<br>"
        f"PIREp ET = {results['eirp_tx_req_dBW']:.2f} dBW<br>"
        f"HPA = {results['hpa_W']:.2f} W"
    )
    rx_text = (
        f"Az = {results['az_rx_deg']:.2f}°<br>El = {results['el_rx_deg']:.2f}°<br>"
        f"Grx = {results['g_rx_dBi']:.2f} dBi<br>"
        f"G/T ET = {results['gt_rx_dB_K']:.2f} dB/K<br>"
        f"PIREp sat = {results['eirp_sat_oper_dBW']:.2f} dBW"
    )
    fig.add_annotation(x=1.8, y=3.10, text=tx_text, showarrow=False, align="center", font=dict(size=13))
    fig.add_annotation(x=10.2, y=3.10, text=rx_text, showarrow=False, align="center", font=dict(size=13))

    conclusion = (
        f"<b>Calidad global</b><br>"
        f"C/N total = {results['cn_total_dB']:.2f} dB &nbsp; | &nbsp; "
        f"C/N mínima = {results['cn_min_dB']:.2f} dB<br>"
        f"Margen = {results['margin_dB']:.2f} dB<br>"
        f"<b style='color:{status_color}'>{results['conclusion_short']}</b>"
    )
    fig.add_shape(type="rect", x0=2.9, y0=0.18, x1=9.1, y1=1.25,
                  line=dict(width=2.5, color=status_color), fillcolor=status_fill, layer="below")
    fig.add_annotation(x=6.0, y=0.72, text=conclusion, showarrow=False, align="center", font=dict(size=15))

    return fig


def create_quality_chart(results: Dict[str, float | str]) -> go.Figure:
    """Gráfica de barras para revisar los resultados C/N."""
    labels = ["C/N subida", "C/N subida sist", "C/N bajada", "C/N bajada sist", "C/N total", "C/N mínima"]
    values = [
        results["cn_up_dB"],
        results["cn_up_sist_dB"],
        results["cn_down_dB"],
        results["cn_down_sist_dB"],
        results["cn_total_dB"],
        results["cn_min_dB"],
    ]
    colors = ["#1E88E5", "#64B5F6", "#7B1FA2", "#BA68C8", "#0B7A3B" if results["valid"] else "#B42318", "#F59E0B"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.2f} dB" for v in values], textposition="outside"))
    fig.add_hline(y=results["cn_min_dB"], line_width=2, line_dash="dash", line_color="#F59E0B",
                  annotation_text="C/N mínima requerida", annotation_position="top left")
    y_min = min(values + [0]) - 2
    y_max = max(values + [results["cn_min_dB"]]) + 5
    fig.update_layout(
        title=dict(text="<b>Comparación de calidad del enlace</b>", x=0.5, xanchor="center"),
        yaxis_title="dB",
        xaxis_title="",
        height=430,
        margin=dict(l=20, r=20, t=60, b=40),
        yaxis=dict(range=[y_min, y_max], gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        showlegend=False,
    )
    return fig

# ================================================================
# Interfaz Streamlit
# ================================================================
st.set_page_config(page_title="SATMEX 5 - Calculadora de enlace", layout="wide")

st.title("Calculadora de enlace satelital SATMEX 5 — versión mejorada")
st.markdown("""
**Materia:** Sistemas de Comunicaciones  
**Integrantes:** Jesús Ademar Santillán Domínguez · Carlos Hernández Pacheco
""")

with st.expander("Fuentes y alcance del programa", expanded=False):
    st.markdown("""
    - Satélite: **SATMEX 5**, posición orbital **116.8° W**.
    - Bandas disponibles: **C** y **Ku**.
    - El plan de frecuencias, polarizaciones, regiones, ATP, back-off y C/I se cargan como datos internos del programa.
    - Los valores de roll-off, FEC, Eb/No, pérdidas y antenas son editables para que el usuario pueda ajustar el diseño.
    - Modo de cálculo: **procedimiento de la guía del proyecto**.
    - Las pérdidas predeterminadas de validación son las del proyecto: apuntamiento = 0.4 dB, polarizador = 0.6 dB, absorción atmosférica = 0.55 dB y lluvia = 0.1 dB.
    - Si quieres usar las pérdidas de tus apuntes anteriores, edita esos campos a 0.3, 0.2, 0.5 y lluvia = 0.0.
    """)

freq_df = create_frequency_plan()

# ------------------ Barra lateral: selección principal ------------------
st.sidebar.header("1) Selección del enlace")
band = st.sidebar.selectbox("Banda", ["Ku", "C"], index=0)
station_names = list(STATION_DATA.keys())

def default_idx(name: str) -> int:
    return station_names.index(name) if name in station_names else 0

et_tx = st.sidebar.selectbox("Estación terrena transmisora", station_names, index=default_idx("MEXICO, D.F."))
et_rx = st.sidebar.selectbox("Estación terrena receptora", station_names, index=default_idx("TIJUANA, BCN."))

mode_tp = st.sidebar.radio("Selección de transpondedor", ["Filtrar por polarización/región", "Elegir transpondedor directamente"], index=1)

available = freq_df[freq_df["band"] == band].copy()
if mode_tp == "Filtrar por polarización/región":
    pol_options = sorted(available["polarization"].unique().tolist())
    pol_sel = st.sidebar.selectbox("Polarización subida/bajada", pol_options, index=0)
    region_options = sorted(available[available["polarization"] == pol_sel]["region"].unique().tolist())
    region_sel = st.sidebar.selectbox("Región", region_options, index=0)
    filtered = available[(available["polarization"] == pol_sel) & (available["region"] == region_sel)].copy()
    tp_sel = st.sidebar.selectbox("Transpondedor", filtered["tp"].tolist())
    tp_row = filtered[filtered["tp"] == tp_sel].iloc[0].to_dict()
else:
    tp_options = available["tp"].tolist()
    default_tp = "12K" if band == "Ku" and "12K" in tp_options else ("12N" if band == "C" and "12N" in tp_options else tp_options[0])
    tp_sel = st.sidebar.selectbox("Transpondedor", tp_options, index=tp_options.index(default_tp))
    tp_row = available[available["tp"] == tp_sel].iloc[0].to_dict()
    pol_sel = tp_row["polarization"]
    region_sel = tp_row["region"]

st.sidebar.header("2) Servicios y modulación")
service_df = st.sidebar.data_editor(
    DEFAULT_SERVICES,
    num_rows="dynamic",
    use_container_width=True,
    key="services_editor",
)
service_df["Total_kbps"] = service_df["Cantidad"].astype(float) * service_df["kbps_por_canal"].astype(float)
rb_kbps = float(service_df["Total_kbps"].sum())
st.sidebar.metric("Rb total", f"{rb_kbps:,.2f} kbps")

modulation = st.sidebar.selectbox("Modulación", list(MODULATION_PRESETS.keys()), index=list(MODULATION_PRESETS.keys()).index("QPSK / 4PSK"))
preset = MODULATION_PRESETS[modulation]

col_a, col_b = st.sidebar.columns(2)
with col_a:
    bits_per_symbol = st.number_input("Bits/símbolo", min_value=1, max_value=8, value=int(preset["bits_per_symbol"]), step=1)
    fec_text = st.text_input("FEC", value=preset["fec"])
with col_b:
    rolloff = st.number_input("Roll-off α", min_value=0.05, max_value=0.50, value=float(preset["rolloff"]), step=0.01)
    ebno_req_dB = st.number_input("Eb/No req. [dB]", min_value=0.0, max_value=30.0, value=float(preset["ebno_dB"]), step=0.1)

try:
    fec_rate = parse_fec(fec_text)
    if not (0 < fec_rate <= 1):
        raise ValueError
except Exception:
    st.sidebar.error("La FEC debe tener formato válido, por ejemplo 3/4 o 0.75.")
    st.stop()

st.sidebar.caption(preset["note"])

st.sidebar.header("3) Antenas y pérdidas")
d_tx_m = st.sidebar.slider("Diámetro antena TX [m]", min_value=0.6, max_value=3.7, value=1.8, step=0.1)
d_rx_m = st.sidebar.slider("Diámetro antena RX [m]", min_value=0.6, max_value=3.7, value=1.8, step=0.1)
eff_tx = st.sidebar.slider("Eficiencia TX", min_value=0.55, max_value=0.70, value=0.70, step=0.01)
eff_rx = st.sidebar.slider("Eficiencia RX", min_value=0.55, max_value=0.70, value=0.70, step=0.01)
tsys_rx_K = st.sidebar.number_input("Temperatura sistema RX [K]", min_value=50.0, max_value=800.0, value=85.0, step=5.0)

loss_pointing_dB = st.sidebar.number_input("Pérdida por apuntamiento [dB]", min_value=0.0, max_value=5.0, value=0.3, step=0.05)
loss_polarizer_dB = st.sidebar.number_input("Pérdida por polarizador [dB]", min_value=0.0, max_value=5.0, value=0.2, step=0.05)
loss_atmos_dB = st.sidebar.number_input("Pérdida por absorción atmosférica [dB]", min_value=0.0, max_value=5.0, value=0.5, step=0.05)
loss_rain_dB = st.sidebar.number_input("Atenuación por lluvia LM [dB]", min_value=0.0, max_value=20.0, value=0.0, step=0.05)
loss_tx_line_dB = st.sidebar.number_input("Pérdida línea TX para HPA [dB]", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
loss_rx_line_dB = st.sidebar.number_input("Pérdida línea RX opcional para G/T ET [dB]", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

if band == "C" and (d_tx_m < 1.8 or d_rx_m < 1.8):
    st.sidebar.warning("Advertencia: en banda C una antena muy pequeña puede ser poco conveniente. Se permite el cálculo, pero revise el margen final.")

st.sidebar.header("4) Operación del transpondedor")
carrier_mode = st.sidebar.radio("Modo", ["Multiportadora", "Una portadora a saturación"], index=0)
params = SAT_PARAMS[band]
ibo_dB = params["IBO_multi_dB"] if carrier_mode == "Multiportadora" else params["IBO_single_dB"]
obo_dB = params["OBO_multi_dB"] if carrier_mode == "Multiportadora" else params["OBO_single_dB"]
use_manual_sat = st.sidebar.checkbox("Editar manualmente EIRP/G/T/SFD", value=False)

# ------------------ Cálculos ------------------
tx = STATION_DATA[et_tx]
rx = STATION_DATA[et_rx]

geo_tx = look_angles(tx["lat"], tx["lon_w"], tx["alt_m"])
geo_rx = look_angles(rx["lat"], rx["lon_w"], rx["alt_m"])

sat_tx_fp = get_sat_footprint(et_tx, band, tp_row["region"], tp_row["polarization"])
sat_rx_fp = get_sat_footprint(et_rx, band, tp_row["region"], tp_row["polarization"])

# Para uplink se usa G/T y SFD en la localidad transmisora; para downlink se usa EIRP de la localidad receptora.
gt_sat_up_dB_K = float(sat_tx_fp["gt"])
sfd_tx_dBW_m2 = float(sat_tx_fp["sfd"])
eirp_sat_sat_dBW = float(sat_rx_fp["eirp"])

if use_manual_sat:
    c1, c2, c3 = st.columns(3)
    with c1:
        gt_sat_up_dB_K = st.number_input("G/T satélite en subida [dB/K]", value=gt_sat_up_dB_K, step=0.1)
    with c2:
        sfd_tx_dBW_m2 = st.number_input("SFD/DFS [dBW/m²]", value=sfd_tx_dBW_m2, step=0.1)
    with c3:
        eirp_sat_sat_dBW = st.number_input("EIRP/PIRE satélite bajada [dBW]", value=eirp_sat_sat_dBW, step=0.1)

symbol_rate_sps, bw_hz = occupied_bandwidth_hz(rb_kbps, bits_per_symbol, fec_rate, rolloff)
bw_MHz = bw_hz / 1e6
ppbw_dB = 10.0 * math.log10(max(bw_hz, 1e-9) / (float(tp_row["bw_MHz"]) * 1e6))

# LU: suma de pérdidas por apuntamiento, polarizador y absorción atmosférica. LM: lluvia.
lu_dB = loss_pointing_dB + loss_polarizer_dB + loss_atmos_dB
lm_dB = loss_rain_dB
link_losses_for_cn_dB = lu_dB + lm_dB

# Ganancias de antena de ET
uplink_GHz = float(tp_row["uplink_GHz"])
downlink_GHz = float(tp_row["downlink_GHz"])
g_tx_dBi = antenna_gain_dbi(d_tx_m, eff_tx, uplink_GHz)
g_rx_dBi = antenna_gain_dbi(d_rx_m, eff_rx, downlink_GHz)
gt_rx_dB_K = g_rx_dBi - loss_rx_line_dB - 10.0 * math.log10(tsys_rx_K)

# === Enlace ascendente, siguiendo la guía ===
d_up_m = geo_tx["distance_km"] * 1000.0
lp_dB = 10.0 * math.log10(4.0 * math.pi * d_up_m ** 2)
eirp_et_sat_dBW = sfd_tx_dBW_m2 + params["ATP_dB"] + lp_dB + ppbw_dB
eirp_tx_req_dBW = eirp_et_sat_dBW - ibo_dB  # PIREp de la estación terrena
hpa_dBW = eirp_tx_req_dBW - g_tx_dBi + loss_tx_line_dB
hpa_W = db_to_linear(hpa_dBW)

lfs_up_dB = free_space_loss_db(geo_tx["distance_km"], uplink_GHz)
cn0_up_dBHz = eirp_tx_req_dBW + gt_sat_up_dB_K - lfs_up_dB - lm_dB - lu_dB + K_BOLTZMANN_DB
cn_up_dB = cn0_up_dBHz - 10.0 * math.log10(bw_hz)

cn_up_sist_dB = inv_sum_db([
    cn_up_dB,
    params["C_I_intermod_up_dB"],
    params["C_I_xpol_up_dB"],
    params["C_I_adj_up_dB"],
])

# === Enlace descendente, siguiendo la guía ===
eirp_sat_oper_dBW = eirp_sat_sat_dBW + ppbw_dB - obo_dB
lfs_down_dB = free_space_loss_db(geo_rx["distance_km"], downlink_GHz)
cn0_down_dBHz = eirp_sat_oper_dBW + gt_rx_dB_K - lfs_down_dB - lm_dB - lu_dB + K_BOLTZMANN_DB
cn_down_dB = cn0_down_dBHz - 10.0 * math.log10(bw_hz)

cn_down_sist_dB = inv_sum_db([
    cn_down_dB,
    params["C_I_intermod_down_dB"],
    params["C_I_xpol_down_dB"],
    params["C_I_adj_down_dB"],
])

# === Calidad global ===
# La guía del proyecto indica combinar C/No_up y C/No_down para el C/No total.
cn0_total_dBHz = inv_sum_db([cn0_up_dBHz, cn0_down_dBHz])
cn_total_dB = cn0_total_dBHz - 10.0 * math.log10(bw_hz)
rb_bps = rb_kbps * 1000.0
cn0_req_dBHz = ebno_req_dB + 10.0 * math.log10(rb_bps)
cn_min_dB = cn0_req_dBHz - 10.0 * math.log10(bw_hz)
margin_dB = cn_total_dB - cn_min_dB
valid = margin_dB >= 0

propagation_s = (geo_tx["distance_km"] + geo_rx["distance_km"]) * 1000.0 / C_LIGHT

# ------------------ Presentación ------------------
cfg = pd.DataFrame([
    ["Satélite", "SATMEX 5"],
    ["Longitud orbital", f"{SATMEX5_LON_W_DEG:.1f}° W"],
    ["Banda", band],
    ["Transpondedor", tp_row["tp"]],
    ["Polarización subida/bajada", tp_row["polarization"]],
    ["Región", tp_row["region"]],
    ["Frecuencia de subida", f"{uplink_GHz:.3f} GHz"],
    ["Frecuencia de bajada", f"{downlink_GHz:.3f} GHz"],
    ["Modulación", modulation],
    ["FEC", fec_text],
    ["Roll-off", f"{rolloff:.2f}"],
    ["Rb", f"{rb_kbps:,.2f} kbps"],
    ["BW ocupado", f"{bw_MHz:.3f} MHz"],
], columns=["Parámetro", "Valor"])

calc_df = pd.DataFrame([
    ["Diferencia longitud TX B", f"{geo_tx['B_deg']:.3f}°"],
    ["Ángulo central TX γ", f"{geo_tx['gamma_deg']:.3f}°"],
    ["Radio estación TX rt", f"{geo_tx['rt_km']:.3f} km"],
    ["Distancia subida", f"{geo_tx['distance_km']:.2f} km"],
    ["Azimut ET transmisora", f"{geo_tx['az_deg']:.2f}°"],
    ["Elevación ET transmisora", f"{geo_tx['el_deg']:.2f}°"],
    ["Diferencia longitud RX B", f"{geo_rx['B_deg']:.3f}°"],
    ["Ángulo central RX γ", f"{geo_rx['gamma_deg']:.3f}°"],
    ["Radio estación RX rt", f"{geo_rx['rt_km']:.3f} km"],
    ["Distancia bajada", f"{geo_rx['distance_km']:.2f} km"],
    ["Azimut ET receptora", f"{geo_rx['az_deg']:.2f}°"],
    ["Elevación ET receptora", f"{geo_rx['el_deg']:.2f}°"],
    ["Distancia total", f"{geo_tx['distance_km'] + geo_rx['distance_km']:.2f} km"],
    ["Tiempo de propagación total", f"{propagation_s:.4f} s"],
    ["Tasa de símbolos", f"{symbol_rate_sps/1e6:.3f} Msym/s"],
    ["Ancho de banda BW", f"{bw_MHz:.3f} MHz"],
    ["C/N0 requerida", f"{cn0_req_dBHz:.3f} dBHz"],
    ["C/N mínima requerida", f"{cn_min_dB:.3f} dB"],
    ["Ganancia antena TX", f"{g_tx_dBi:.2f} dBi"],
    ["Ganancia antena RX", f"{g_rx_dBi:.2f} dBi"],
    ["G/T estación receptora", f"{gt_rx_dB_K:.2f} dB/K"],
    ["SFD usado", f"{sfd_tx_dBW_m2:.2f} dBW/m²"],
    ["G/T satélite usado", f"{gt_sat_up_dB_K:.2f} dB/K"],
    ["PIRE satélite saturación/mapa", f"{eirp_sat_sat_dBW:.2f} dBW"],
    ["ATP", f"{params['ATP_dB']:.2f} dB"],
    ["BOI / IBO", f"{ibo_dB:.2f} dB"],
    ["BOO / OBO", f"{obo_dB:.2f} dB"],
    ["LU = Lap + Lpol + Latm", f"{lu_dB:.2f} dB"],
    ["LM lluvia", f"{lm_dB:.2f} dB"],
    ["Pérdida espacio libre subida LFS", f"{lfs_up_dB:.2f} dB"],
    ["Pérdida espacio libre bajada LFS", f"{lfs_down_dB:.2f} dB"],
    ["LP dispersión subida", f"{lp_dB:.3f} dB"],
    ["PPBW", f"{ppbw_dB:.3f} dB"],
    ["PIRE ET antes de BOI", f"{eirp_et_sat_dBW:.3f} dBW"],
    ["PIREp ET después de BOI", f"{eirp_tx_req_dBW:.3f} dBW"],
    ["Potencia HPA calculada", f"{hpa_W:.2f} W ({hpa_dBW:.2f} dBW)"],
    ["C/N0 subida", f"{cn0_up_dBHz:.3f} dBHz"],
    ["C/N subida", f"{cn_up_dB:.3f} dB"],
    ["C/N subida del sistema", f"{cn_up_sist_dB:.3f} dB"],
    ["PIREp satélite", f"{eirp_sat_oper_dBW:.3f} dBW"],
    ["C/N0 bajada", f"{cn0_down_dBHz:.3f} dBHz"],
    ["C/N bajada", f"{cn_down_dB:.3f} dB"],
    ["C/N bajada del sistema", f"{cn_down_sist_dB:.3f} dB"],
    ["C/N0 total del sistema", f"{cn0_total_dBHz:.3f} dBHz"],
    ["C/N total global", f"{cn_total_dB:.3f} dB"],
    ["Margen del enlace", f"{margin_dB:.2f} dB"],
], columns=["Cálculo", "Resultado"])

results = {
    "tx_name": et_tx,
    "rx_name": et_rx,
    "uplink_GHz": uplink_GHz,
    "downlink_GHz": downlink_GHz,
    "d_up_km": geo_tx["distance_km"],
    "d_down_km": geo_rx["distance_km"],
    "az_tx_deg": geo_tx["az_deg"],
    "el_tx_deg": geo_tx["el_deg"],
    "az_rx_deg": geo_rx["az_deg"],
    "el_rx_deg": geo_rx["el_deg"],
    "g_tx_dBi": g_tx_dBi,
    "g_rx_dBi": g_rx_dBi,
    "gt_rx_dB_K": gt_rx_dB_K,
    "eirp_tx_req_dBW": eirp_tx_req_dBW,
    "eirp_sat_oper_dBW": eirp_sat_oper_dBW,
    "hpa_W": hpa_W,
    "lfs_up_dB": lfs_up_dB,
    "lfs_down_dB": lfs_down_dB,
    "cn_up_dB": cn_up_dB,
    "cn_up_sist_dB": cn_up_sist_dB,
    "cn_down_dB": cn_down_dB,
    "cn_down_sist_dB": cn_down_sist_dB,
    "cn_total_dB": cn_total_dB,
    "cn_min_dB": cn_min_dB,
    "margin_dB": margin_dB,
    "valid": valid,
    "conclusion_short": "ENLACE VÁLIDO" if valid else "ENLACE NO VÁLIDO",
}

# Métricas superiores
m1, m2, m3, m4 = st.columns(4)
m1.metric("BW ocupado", f"{bw_MHz:.3f} MHz")
m2.metric("C/N total", f"{cn_total_dB:.2f} dB")
m3.metric("C/N mínima", f"{cn_min_dB:.2f} dB")
m4.metric("Margen", f"{margin_dB:.2f} dB")

if valid:
    st.success("El enlace satisface la relación portadora a ruido mínima requerida; por lo tanto, el enlace es válido.")
else:
    st.error("El enlace no satisface la relación portadora a ruido mínima requerida; se debe ajustar el diseño.")

resumen_tab, datos_tab, calculos_tab, diagrama_tab, reporte_tab = st.tabs([
    "📌 Resumen", "📡 Datos del enlace", "🧮 Paso a paso", "📊 Diagrama y gráfica", "⬇️ Reporte"
])

with resumen_tab:
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown("### Configuración seleccionada")
        st.dataframe(cfg, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### Resultado final")
        status = "✅ Cumple" if valid else "❌ No cumple"
        summary_df = pd.DataFrame([
            ["C/N total del sistema", f"{cn_total_dB:.3f} dB"],
            ["C/N mínima requerida", f"{cn_min_dB:.3f} dB"],
            ["Margen", f"{margin_dB:.3f} dB"],
            ["Validación", status],
        ], columns=["Parámetro", "Resultado"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.markdown("### Conclusión")
        if valid:
            st.info(f"Como C/N total = {cn_total_dB:.2f} dB es mayor que C/N mínima = {cn_min_dB:.2f} dB, el enlace satisface la relación portadora a ruido mínima requerida; por lo tanto, el enlace es válido.")
        else:
            st.warning(f"Como C/N total = {cn_total_dB:.2f} dB no supera a C/N mínima = {cn_min_dB:.2f} dB, el enlace no es válido bajo estas condiciones.")

with datos_tab:
    st.markdown("### Servicios considerados")
    st.dataframe(service_df, use_container_width=True, hide_index=True)
    st.markdown("### Datos cargados desde SATMEX 5 / configuración técnica")
    sat_df = pd.DataFrame([
        ["SFD en estación transmisora", f"{sfd_tx_dBW_m2:.2f} dBW/m²"],
        ["G/T satélite para subida", f"{gt_sat_up_dB_K:.2f} dB/K"],
        ["PIRE satélite para bajada", f"{eirp_sat_sat_dBW:.2f} dBW"],
        ["ATP", f"{params['ATP_dB']:.2f} dB"],
        ["BOI usado", f"{ibo_dB:.2f} dB"],
        ["BOO usado", f"{obo_dB:.2f} dB"],
        ["C/I intermod subida", f"{params['C_I_intermod_up_dB']:.2f} dB"],
        ["C/I x-pol subida", f"{params['C_I_xpol_up_dB']:.2f} dB"],
        ["C/X sat ady subida", f"{params['C_I_adj_up_dB']:.2f} dB"],
        ["C/I intermod bajada", f"{params['C_I_intermod_down_dB']:.2f} dB"],
        ["C/I x-pol bajada", f"{params['C_I_xpol_down_dB']:.2f} dB"],
        ["C/X sat ady bajada", f"{params['C_I_adj_down_dB']:.2f} dB"],
    ], columns=["Dato", "Valor"])
    st.dataframe(sat_df, use_container_width=True, hide_index=True)

with calculos_tab:
    st.markdown("### Cálculos principales del procedimiento")
    st.dataframe(calc_df, use_container_width=True, hide_index=True)
    with st.expander("Ecuaciones usadas", expanded=False):
        st.latex(r"BW=R_b\left(\frac{1}{r}\right)\left(\frac{1}{MF}\right)(1+\beta)")
        st.latex(r"[PIRE]_{ET}=SFD+ATP+L_P+PPBW")
        st.latex(r"[PIREp]_{ET}=[PIRE]_{ET}-BOI")
        st.latex(r"\left[\frac{C}{N_0}\right]_{up}=[PIREp]_{ET}+\left[\frac{G}{T}\right]_{sat}-L_{FS}-L_M-L_U-k")
        st.latex(r"[PIREp]_{sat}=[PIRE]_{sat}+PPBW-BOO")
        st.latex(r"\left[\frac{C}{N_0}\right]_{down}=[PIREp]_{sat}+\left[\frac{G}{T}\right]_{ET}-L_{FS}-L_M-L_U-k")
        st.latex(r"\left[\frac{C}{N_0}\right]_{tot}=10\log_{10}\left(\frac{1}{10^{-(C/N_0)_{up}/10}+10^{-(C/N_0)_{down}/10}}\right)")

with diagrama_tab:
    st.plotly_chart(create_diagram(results), use_container_width=True)
    st.plotly_chart(create_quality_chart(results), use_container_width=True)

conclusion = (
    f"La relación portadora a ruido total del sistema es C/N total = {cn_total_dB:.2f} dB, "
    f"mientras que la relación mínima requerida es C/N mínima = {cn_min_dB:.2f} dB. "
    f"El margen calculado es {margin_dB:.2f} dB. "
    + ("Como C/N total es mayor que C/N mínima, el enlace satisface la relación portadora a ruido mínima requerida; por lo tanto, el enlace es válido."
       if valid else
       "Como C/N total no supera la relación mínima requerida, el enlace no es válido bajo las condiciones actuales.")
)

report_lines = [
    "# Reporte de enlace satelital SATMEX 5",
    "",
    "**Materia:** Sistemas de Comunicaciones",
    "**Integrantes:** Jesús Ademar Santillán Domínguez y Carlos Hernández Pacheco",
    "",
    "## Configuración",
    cfg.to_markdown(index=False),
    "",
    "## Servicios",
    service_df.to_markdown(index=False),
    "",
    "## Resultados",
    calc_df.to_markdown(index=False),
    "",
    "## Conclusión",
    conclusion,
]
report_md = "\n".join(report_lines)

with reporte_tab:
    st.markdown("### Conclusión automática")
    st.info(conclusion)
    st.download_button(
        "Descargar reporte en Markdown",
        data=report_md.encode("utf-8"),
        file_name="reporte_enlace_satmex5.md",
        mime="text/markdown",
    )
