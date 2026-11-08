# Guía teórica — Estudio de caso: exploración y calidad de datos

*Apoyo para el notebook `EIARV011_A2_analisis_estudiante.ipynb` (dataset `StudentsPerformance.csv`) y para el informe final de la actividad. Léela antes de empezar y vuelve a ella cuando redactes las respuestas.*

---

## 1. Ruta de trabajo

La actividad sigue el ciclo básico de un proyecto de ciencia de datos:

```
Cargar → Entender variables → Revisar calidad → Explorar (EDA) → Relacionar variables → Concluir
```

| Paso | Pregunta que responde | Sección del notebook |
|---|---|---|
| Cargar | ¿Pude leer el archivo correctamente? | 1 |
| Entender variables | ¿Qué hay en cada columna y de qué tipo es? | 2 |
| Calidad | ¿Puedo confiar en estos datos? | 3 |
| EDA | ¿Cómo se distribuye cada variable? | 4 |
| Relaciones | ¿Qué variables se asocian entre sí? | 5 |
| Concluir | ¿Qué diría en el informe? | 6 |

**Regla de oro:** no se analiza ni se concluye nada sin revisar antes la calidad de los datos.

## 2. Comprender las variables

Lo primero es saber **cuántos datos hay y qué es cada columna**.

| Instrucción | Qué te dice |
|---|---|
| `df.shape` | Número de filas y de columnas |
| `df.columns` | Nombres de las columnas |
| `df.dtypes` | Tipo de dato de cada columna |
| `df.info()` | Resumen de todo lo anterior y conteo de datos no nulos |

**Tipos de dato en pandas**

- `object` → texto. Casi siempre es una variable **categórica**.
- `int64` → número entero. `float64` → número decimal. Son variables **numéricas**.

**Tipos de variable**

| | Qué representa | Ejemplo | Cómo se describe |
|---|---|---|---|
| **Categórica nominal** | Grupos sin orden | género, tipo de almuerzo | Frecuencias y porcentajes |
| **Categórica ordinal** | Grupos con orden natural | nivel educativo (de menor a mayor) | Frecuencias; el orden importa |
| **Numérica** | Cantidades medibles | puntajes | Media, mediana, desviación, mínimo, máximo |

> Un puntaje de 0 a 100 es numérico aunque solo tome valores enteros (numérica discreta).

## 3. Calidad de datos

Los datos reales casi nunca llegan perfectos. Se revisan tres problemas típicos:

### 3.1 Valores nulos
Datos faltantes (`NaN`). `df.isnull()` marca con `True` cada celda vacía y `.sum()` los cuenta por columna.

Posibles decisiones: eliminar las filas, imputar (rellenar con la media, la mediana o la moda) o dejarlos y documentarlo. Depende de **cuántos** faltan y **por qué**.

### 3.2 Duplicados
Filas idénticas repetidas. `df.duplicated().sum()` las cuenta. Distorsionan frecuencias y promedios porque un mismo caso se cuenta más de una vez.

### 3.3 Valores atípicos (*outliers*)
Valores que se alejan mucho del resto. Un método estándar es el **rango intercuartílico (IQR)**:

1. **Q1** = percentil 25 y **Q3** = percentil 75.
2. **IQR = Q3 − Q1** (el ancho del 50 % central de los datos).
3. Límite inferior = **Q1 − 1,5 × IQR**.
4. Límite superior = **Q3 + 1,5 × IQR**.
5. Es atípico todo valor **fuera** de esos límites.

**Cuidados al interpretarlo**
- **Atípico ≠ error.** Puede ser un caso real poco común (un estudiante que de verdad sacó muy mala nota) o un dato mal digitado.
- Antes de eliminar, **verifica con la fuente** y documenta la decisión. Eliminar sin criterio puede ocultar información.
- Revisa si los límites son alcanzables: si la escala es 0–100, un límite superior mayor que 100 significa que no puede haber atípicos hacia arriba.

### 3.4 Boxplot (diagrama de caja)

```
  bigote ├──────┤ caja │  mediana │ ├──────┤ bigote      ●  ← atípico
                Q1     Q2        Q3
```

- La **caja** va de Q1 a Q3 (50 % central).
- La **línea interna** es la mediana.
- Los **bigotes** llegan hasta los datos que aún están dentro de los límites del IQR.
- Los **puntos aislados** son los atípicos.

Es la versión visual del cálculo con IQR: ambos deben coincidir.

### 3.5 Otras revisiones de calidad
- **Valores fuera de rango:** por ejemplo, un puntaje de 120 en una escala de 0 a 100.
- **Inconsistencias en el formato:** nombres de columnas con espacios o caracteres especiales (`race/ethnicity`) que dificultan el código. Se puede normalizar a *snake_case*.

## 4. Análisis exploratorio (EDA)

### 4.1 Variables categóricas
Se **cuentan**, no se promedian.
- **Frecuencia absoluta:** `value_counts()`.
- **Frecuencia relativa (%):** `value_counts(normalize=True) * 100`.
- Gráfico adecuado: **barras**.

**Balance:** una variable está *balanceada* si sus categorías tienen tamaños parecidos. Si un grupo es muy pequeño, las comparaciones con él son poco confiables.

### 4.2 Variables numéricas
`describe()` entrega `count`, `mean`, `std`, `min`, `25%`, `50%`, `75%` y `max`.

- **Media:** el promedio; sensible a valores extremos.
- **Mediana (`50%`):** el valor central; casi no la afectan los extremos.
- **Desviación estándar:** cuánto se alejan los datos de la media en promedio.
- **Cuartiles:** dividen los datos ordenados en cuatro partes iguales.

**Pista sobre la forma:** media ≈ mediana → distribución aproximadamente simétrica; media > mediana → cola hacia valores altos; media < mediana → cola hacia valores bajos.

### 4.3 Histograma
Agrupa los valores en rangos (*bins*) y dibuja una barra con la cantidad de casos por rango. Qué mirar: dónde está el pico, si es simétrico o tiene cola, y si se parece a una campana (distribución normal).

## 5. Relaciones entre variables

### 5.1 Correlación (numérica vs. numérica)
`df[[...]].corr()` mide **qué tan relacionadas** están dos variables numéricas, con un valor entre **−1 y 1**:

| Valor de r | Interpretación aproximada |
|---|---|
| Cercano a **1** | Relación positiva fuerte: si una sube, la otra tiende a subir |
| Cercano a **0** | Casi sin relación lineal |
| Cercano a **−1** | Relación negativa fuerte: si una sube, la otra tiende a bajar |

- La diagonal siempre vale 1 (cada variable consigo misma).
- Lo común es considerar |r| > 0,7 como fuerte y |r| < 0,3 como débil, aunque los límites son orientativos.

### 5.2 Comparar grupos (categórica vs. numérica)
`groupby` divide los datos por categoría y calcula una medida en cada grupo:

```python
df.groupby("categoría")["variable_numérica"].mean()
```

Si los promedios de los grupos son muy distintos, hay una **asociación** entre la categoría y la variable numérica.

### 5.3 Variables derivadas
Se pueden crear columnas nuevas a partir de otras. Por ejemplo, un promedio por fila:

```python
df["promedio"] = df[["col1", "col2", "col3"]].mean(axis=1)
```

`axis=1` significa "calcula por fila" (por estudiante); `axis=0` sería por columna.

### 5.4 ⚠️ Correlación y asociación no son causalidad
Que dos cosas aparezcan juntas **no prueba que una cause la otra**. Pueden intervenir factores que no están en la tabla (motivación, recursos, contexto familiar). Por eso en el informe se escriben frases como "se asocia con" o "tiende a", y no "causa" o "produce".

## 6. Cómo redactar el informe

Cada afirmación del informe debe cumplir tres cosas:

1. **Estar respaldada por un dato** del notebook (un número, una tabla o un gráfico).
2. **Interpretar**, no solo copiar el número: qué significa y por qué importa.
3. **Cuidar el lenguaje**: asociación, no causalidad; hallazgo, no certeza.

Guía por pregunta:

| # | Pregunta | Cómo abordarla |
|---|---|---|
| 1 | Fenómeno que representa el dataset | Describe quién, qué se mide y en qué contexto, en dos o tres frases |
| 2 | 5 variables más importantes | Elige y **justifica con datos** (relación con el resultado, utilidad práctica); no basta con enumerarlas |
| 3 | Distribución de variables | Una observación para las categóricas (balance) y otra para las numéricas (centro, dispersión, forma) |
| 4 | Al menos 2 relaciones | Cita el valor (r o diferencia de promedios) y di en qué dirección va la relación |
| 5 | Problemas de calidad | Qué encontraste, cuántos casos, y **qué harías** y por qué |
| 6 | Pregunta de IA a futuro | Define qué quieres predecir o descubrir y el tipo de problema (ver abajo) |

### Tipos de problema de IA (para la pregunta 6)

| Tipo | Qué hace | Ejemplo genérico |
|---|---|---|
| **Regresión** | Predice un número | Estimar una nota a partir de otras variables |
| **Clasificación** | Predice una categoría | Aprobado / no aprobado |
| **Clustering** | Agrupa casos parecidos sin etiquetas | Segmentar perfiles de estudiantes |

Pregunta clave: ¿lo que quiero predecir es un **número**, una **categoría**, o quiero **descubrir grupos**?

## 7. Errores comunes

- Escribir `df.math score`: los nombres con espacios van entre comillas, `df["math score"]`.
- `FileNotFoundError`: el CSV debe estar en la misma carpeta que el notebook.
- Calcular promedios de variables categóricas.
- Eliminar atípicos automáticamente, sin verificar si son errores.
- Confundir `50%` con "el 50 % de los estudiantes": es la **mediana**.
- Afirmar causalidad a partir de una correlación o una diferencia de promedios.
- Dar números en el informe sin interpretarlos, o interpretaciones sin ningún número.
- Elegir variables "importantes" sin explicar por qué.

## 8. Mini-glosario

| Término | Significado corto |
|---|---|
| Dataset / DataFrame | Tabla de datos / su representación en pandas |
| Variable categórica / numérica | Grupos / cantidades |
| Nulo (`NaN`) | Dato faltante |
| Duplicado | Fila repetida |
| Atípico (*outlier*) | Valor muy alejado del resto |
| Cuartil / IQR | Divide los datos en cuatro partes / ancho del 50 % central |
| Boxplot | Gráfico de caja: mediana, cuartiles y atípicos |
| Histograma | Barras por rangos de una variable numérica |
| Correlación (r) | Fuerza de relación entre dos numéricas, de −1 a 1 |
| `groupby` | Agrupar por categoría y calcular una medida |
| Variable derivada | Columna nueva construida a partir de otras |
| Regresión / clasificación / clustering | Predecir un número / una categoría / descubrir grupos |
