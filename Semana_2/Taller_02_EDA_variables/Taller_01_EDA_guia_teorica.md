# Guía teórica — Análisis exploratorio de variables categóricas y numéricas

*Apoyo para el taller `Taller_01_EDA.md`. Léela antes de empezar los ejercicios.*

---

## 1. ¿Qué es el EDA y por qué se hace primero?

El **Análisis Exploratorio de Datos (EDA)** consiste en "conocer" un conjunto de datos antes de usarlo para algo más complejo (por ejemplo, entrenar un modelo de IA). Se hace con resúmenes numéricos y gráficos sencillos.

> Analogía: antes de cocinar, revisas qué ingredientes tienes y en qué estado están.

Este taller es **descriptivo**: nos dice *qué hay* en los datos y *cómo se distribuyen*, no *por qué* ocurre.

## 2. Estructura de un dataset

| Concepto | Qué es | En `StudentsPerformance.csv` |
|---|---|---|
| **Fila** (registro) | Un caso observado | Un estudiante |
| **Columna** (variable) | Una característica medida | `gender`, `math score`, etc. |
| **DataFrame** | La tabla cargada en pandas | `df` |

## 3. Los dos tipos de variables

| | **Categórica** | **Numérica** |
|---|---|---|
| Qué representa | Un grupo o etiqueta | Una cantidad medible |
| Ejemplo | `gender`, `lunch` | `math score` |
| ¿Tiene sentido promediar? | **No** (¿cuál es el "promedio" de `female` y `male`?) | **Sí** |
| Cómo se describe | **Frecuencias** y porcentajes | **Media, mediana, desviación, mínimo, máximo** |
| Gráfico adecuado | **Barras** | **Histograma** (o boxplot) |

**Regla práctica:** primero identifica el tipo de variable; eso decide qué herramienta usar.

## 4. Variables categóricas: contar

- **Frecuencia absoluta:** cuántos casos hay en cada categoría (`value_counts()`).
- **Frecuencia relativa (porcentaje):** qué parte del total representa cada categoría (`value_counts(normalize=True) * 100`).

**Balance:** una variable está *balanceada* si sus categorías tienen tamaños parecidos. Importa porque, si un grupo es muy pequeño, las comparaciones que hagas con él son menos confiables. Ejemplo: comparar los promedios de un grupo de 900 personas con otro de 100.

## 5. Variables numéricas: resumir

Un resumen estadístico responde dos preguntas: **¿dónde está el centro?** y **¿qué tan dispersos están los datos?**

**Medidas de posición (el centro)**

- **Media:** la suma de los valores dividida entre la cantidad. Es sensible a valores extremos.
- **Mediana (`50%`):** el valor del medio cuando se ordenan los datos; la mitad queda por debajo y la mitad por encima. Casi no la afectan los extremos.
- **Cuartiles (`25%`, `50%`, `75%`):** dividen los datos ordenados en cuatro partes iguales. Entre el `25%` y el `75%` está el 50 % central.

**Medidas de dispersión**

- **Desviación estándar (`std`):** cuánto se alejan los datos de la media, en promedio. Más `std` = datos más dispersos.
- **Mínimo y máximo:** los extremos. **Rango = máximo − mínimo.**

`describe()` entrega todo esto de una vez: `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`.

## 6. Media vs. mediana: una pista sobre la forma

- **Media ≈ mediana** → la distribución es aproximadamente **simétrica**.
- **Media > mediana** → hay valores muy altos que "jalan" la media hacia arriba (cola a la derecha).
- **Media < mediana** → hay valores muy bajos que "jalan" la media hacia abajo (cola a la izquierda).

## 7. El histograma y la forma de una distribución

Un **histograma** agrupa los valores en rangos (*bins*) y dibuja una barra con la cantidad de casos de cada rango. Más `bins` = más detalle, pero también más ruido.

Cosas que debes mirar:

1. **Dónde está el pico:** el valor más frecuente.
2. **Simetría:** ¿cae igual hacia los dos lados o tiene una **cola** más larga hacia uno? (asimetría)
3. **Forma de campana** (distribución normal): pico al centro y caída simétrica. Es común en variables como notas o estaturas.

## 8. Comparar grupos con `groupby` (reto final)

`groupby` divide los datos por categoría y calcula una medida en cada grupo:

```python
df.groupby("categoría")["variable_numérica"].mean()
```

Interpretación: si los promedios de los grupos son muy distintos, **hay una asociación** entre la categoría y la variable numérica.

⚠️ **Correlación no es causalidad.** Que dos cosas aparezcan juntas no prueba que una cause la otra: pueden influir otros factores que no están en la tabla. Por eso la pregunta de cierre pide una **hipótesis**, no una conclusión definitiva.

## 9. Errores comunes

- Calcular promedios de variables categóricas.
- Escribir `df.math score`: los nombres con espacios van entre comillas, `df["math score"]`.
- Olvidar `ax=axes[i]` al graficar con `plt.subplots`: salen figuras separadas en lugar de un panel.
- Leer `50%` como "el 50 % de los estudiantes": es la **mediana**.
- Responder las preguntas de reflexión solo con "sí" o "no": hay que justificar con un número o con lo que se ve en el gráfico.

## 10. Mini-glosario

| Término | Significado corto |
|---|---|
| Dataset | Conjunto de datos organizado en filas y columnas |
| Variable | Característica medida (columna) |
| Frecuencia | Cuántas veces aparece un valor o categoría |
| Media | Promedio |
| Mediana | Valor central |
| Desviación estándar | Dispersión respecto a la media |
| Cuartil | Divide los datos ordenados en cuatro partes |
| Histograma | Gráfico de barras por rangos de una variable numérica |
| Asimetría | Cola más larga hacia un lado |
| `groupby` | Agrupar por categoría y calcular una medida |
