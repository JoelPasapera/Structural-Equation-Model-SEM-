"""
Este script utiliza la biblioteca rpy2 para realizar un análisis SEM
(Structural Equation Modeling) en Python utilizando el paquete lavaan de R.
Asegúrate de tener R y el paquete lavaan instalados en tu sistema.
"""

import os
import rpy2.situation

# Debe mostrar la ruta correcta
right_path = rpy2.situation.get_r_home()

# Ajusta la ruta
os.environ["R_HOME"] = right_path
import numpy as np
import pandas as pd
import rpy2.robjects as ro  # usar despues para codigo mas limpio (Ignore-unused)
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import (
    FloatVector,
)  # usar despues para codigo mas limpio (Ignore-unused)


pandas2ri.activate()

# Cargar paquetes R
lavaan = importr("lavaan")
base = importr("base")

# ------------------------------------------
# Generar datos con estructura real
# ------------------------------------------
np.random.seed(123)
n = 500  # Muestra grande para estabilidad

# Factores latentes
Satisfaccion = np.random.normal(0, 1, n)
Lealtad = 0.6 * Satisfaccion + np.random.normal(0, 0.5, n)

# Variables observadas
data = pd.DataFrame(
    {
        "sat1": Satisfaccion * 0.8 + np.random.normal(0, 0.5, n),
        "sat2": Satisfaccion * 0.7 + np.random.normal(0, 0.6, n),
        "sat3": Satisfaccion * 0.9 + np.random.normal(0, 0.4, n),
        "leal1": Lealtad * 0.7 + np.random.normal(0, 0.5, n),
        "leal2": Lealtad * 0.8 + np.random.normal(0, 0.6, n),
    }
)

# Escalar y redondear
data = data.apply(
    lambda x: np.round((x - x.min()) / (x.max() - x.min()) * 4 + 1)  # Escala 1-5
)

# ------------------------------------------
# Modelo SEM
# ------------------------------------------
model = """
    # Modelo de medición
    Satisfaccion =~ sat1 + sat2 + sat3
    Lealtad =~ leal1 + leal2

    # Modelo estructural (agregamos intercepto)
    Lealtad ~ a*Satisfaccion
    
    # Fijar varianza de factores
    Satisfaccion ~~ start(0.1)*Satisfaccion
    Lealtad ~~ start(0.1)*Lealtad
"""

# Convertir datos a R
data_r = pandas2ri.py2rpy(data)

# Ajustar modelo
fit = lavaan.sem(
    model, data=data_r, estimator="ML", bounds=True  # Prevenir varianzas negativas
)

# ------------------------------------------
# Resultados
# ------------------------------------------
# Resumen básico
print(lavaan.summary(fit, standardized=True))


# ... todo tu setup previo, datos, model, fit = lavaan.sem(...)
# Ahora, extraemos fitMeasures SIN auto-conversión:

with localconverter(default_converter):
    fit_measures_r = lavaan.fitMeasures(fit)  # sigue siendo un R vector

# Extraemos nombres y valores
measure_names = list(fit_measures_r.names)  # atributo .names
measure_values = list(fit_measures_r)  # valores numéricos

# Armamos el DataFrame
fit_measures_df = pd.DataFrame(
    {"measure": measure_names, "value": measure_values}
).set_index("measure")

print(fit_measures_df.loc[["cfi", "rmsea", "srmr"], :])

