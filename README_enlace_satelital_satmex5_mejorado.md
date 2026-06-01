# Calculadora de enlace satelital SATMEX 5 — versión mejorada

Materia: Sistemas de Comunicaciones  
Integrantes: Jesús Ademar Santillán Domínguez y Carlos Hernández Pacheco

## Cómo ejecutar

```powershell
python -m pip install -r requirements_satmex5_mejorado.txt
python -m streamlit run app_enlace_satelital_satmex5_mejorado.py
```

## Cambios principales

- Interfaz reorganizada con pestañas: resumen, datos, paso a paso, diagrama/gráfica y reporte.
- Caso predeterminado: CDMX → Tijuana, banda Ku, transpondedor 12K.
- Se conserva el cálculo de azimut como estaba en el procedimiento usado previamente.
- Las pérdidas siguen siendo editables desde la barra lateral.
- Se agregó una gráfica de comparación de C/N y un diagrama final más claro.
- No se agregó un modo libreta; todos los datos se pueden cambiar directamente.
