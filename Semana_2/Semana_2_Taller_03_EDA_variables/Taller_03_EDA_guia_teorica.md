# Guía teórica — Construir y analizar variables a partir de conteos

*Apoyo para el taller `Taller_03_EDA.ipynb` (dataset `credits.csv`). Léela antes de empezar los ejercicios. Si no recuerdas qué es una variable categórica o numérica, repasa la guía del Taller 02.*

---

## 1. La idea central de este taller

En el Taller 02 el dataset ya traía columnas numéricas (las notas). Aquí **no**: `credits.csv` solo tiene identificadores y texto. Entonces aprenderás algo muy común en el trabajo real con datos:

> **Si no tienes la variable numérica que necesitas, la construyes a partir de conteos.**

Con ese conteo (el "tamaño del reparto" de cada película) harás después el mismo análisis de siempre: `describe()` e histogramas.

## 2. ¿Qué representa cada fila?

Antes de contar nada, pregúntate: **¿qué es una fila?**

| Columna | Significado |
|---|---|
| `person_id` | Identificador de la persona |
| `id` | Identificador del título (película/serie) |
| `name` | Nombre de la persona |
| `character` | Personaje (solo actores) |
| `role` | `ACTOR` o `DIRECTOR` |

Cada fila es **un crédito**: una persona acreditada en un título. No es una película ni una persona. Por eso una película aparece en muchas filas y una misma persona puede aparecer en varias.

> Los porcentajes de `role` son "porcentaje de **créditos**", no de personas ni de películas. Siempre fíjate en la unidad.

## 3. Categóricas con muchas categorías (alta cardinalidad)

- **Cardinalidad** = cuántos valores distintos tiene una variable.
- `role` tiene 2 valores (baja cardinalidad); `name` tiene decenas de miles (alta).
- Con alta cardinalidad no tiene sentido mostrar todas las categorías. Se resume con las **más frecuentes**: `value_counts().head(10)`.

⚠️ El top 10 muestra los casos más extremos, **no lo típico**. En este dataset, la gran mayoría de personas aparece una sola vez.

## 4. Desbalance: ¿problema o característica?

En el Taller 02 vimos que un grupo muy pequeño da comparaciones poco confiables. Aquí `role` también está muy desbalanceada (casi todo son `ACTOR`), pero la causa es distinta:

| | Taller 02 (`test preparation course`) | Taller 03 (`role`) |
|---|---|---|
| Causa del desbalance | La muestra de estudiantes | La **naturaleza del fenómeno** |
| ¿Hay que preocuparse? | Sí, al comparar grupos | No: una película tiene muchos actores y normalmente un director |

**Pregunta útil:** ¿el desbalance indica un problema de los datos o simplemente cómo funciona el mundo?

## 5. Construir una variable numérica con `groupby` + `size`

Receta de tres pasos:

```python
# 1) Filtrar las filas que interesan
actores = df[df["role"] == "ACTOR"]

# 2) Agrupar por película y contar filas de cada grupo
cast_size = actores.groupby("id").size()

# 3) Resumir con las herramientas de siempre
cast_size.describe()
```

Cómo pensarlo:

1. **Filtrar:** para contar solo actores, primero te quedas con las filas de actores.
2. **Agrupar:** `groupby("id")` forma un grupo por película.
3. **Contar:** `size()` devuelve cuántas filas tiene cada grupo. El resultado tiene **un número por película**: ya es una variable numérica.

Dos detalles:
- Una película sin ningún crédito de actor no aparece en `cast_size` (no tiene filas tras el filtro).
- `cast_size` es una Serie cuyo índice es `id`; para ordenarla usa `sort_values(ascending=False)`.

## 6. Distribución en campana vs. distribución sesgada

Con las notas del Taller 02, la media y la mediana eran casi iguales y el histograma parecía una campana. Con el tamaño del reparto **no será así**.

| | **Simétrica (campana)** | **Sesgada a la derecha (cola larga)** |
|---|---|---|
| Media vs. mediana | Parecidas | **Media > mediana** |
| Histograma | Pico al centro, cae igual a ambos lados | Pico a la izquierda y una cola larga hacia la derecha |
| Ejemplo típico | Notas, estaturas | Ingresos, seguidores en redes, ventas, tamaño de repartos |
| Por qué ocurre | Muchos factores pequeños se compensan | Hay un mínimo natural y unos pocos casos enormes |

**Por qué la media sube:** la media suma todos los valores, así que unos pocos valores gigantes la "jalan" hacia arriba. La mediana solo mira el valor del medio, y casi no se mueve.

**Moraleja:** mira siempre el histograma antes de asumir la forma de una variable. Y si hay sesgo, la **mediana** suele describir mejor lo "típico" que la media.

## 7. Comparar dispersión entre variables

La desviación estándar (`std`) dice qué tan variable es una cantidad. Al comparar `cast_size` con `director_size`:

- Si casi todas las películas tienen **exactamente un director**, casi no hay variación → `std` muy pequeña.
- El número de actores cambia mucho de una película a otra → `std` grande.

Una `std` pequeña no significa "mejor" ni "peor": significa que los valores son casi iguales entre sí. Conviene compararla junto con la media, porque 5 de desviación es muy distinto si la media es 10 que si es 10 000.

## 8. Patrón descriptivo vs. causa

El reto final pide las películas con mayor reparto y una hipótesis que lo explique. Ten presente:

- El dataset **describe un patrón** (cuántas personas fueron acreditadas).
- **No explica por qué.** Un reparto grande puede deberse al género, al presupuesto, a cómo cada estudio acredita o a qué tan completa es la fuente. Eso son **hipótesis**, y para confirmarlas harían falta otros datos.

Una buena hipótesis es la que podrías comprobar con datos adicionales.

## 9. Errores comunes

- Olvidar filtrar `role == "ACTOR"` antes de agrupar: se mezclan directores y actores.
- Contar sobre el DataFrame completo en vez de contar por `id`: obtendrías el total de créditos, no el de cada película.
- Interpretar los porcentajes de `role` como porcentaje de películas o de personas.
- Esperar una campana: aquí la forma es distinta y eso **es parte de lo que debes observar**.
- Olvidar `ax=axes[i]` en `plt.subplots`: salen figuras separadas.
- Responder solo "sí/no" en las reflexiones: justifica con un número (por ejemplo, media vs. mediana) o con lo que ves en el gráfico.

## 10. Mini-glosario

| Término | Significado corto |
|---|---|
| Crédito | Una fila: una persona acreditada en un título |
| Cardinalidad | Número de valores distintos de una variable |
| Variable derivada | Variable que construyes a partir de otras (p. ej. `cast_size`) |
| `groupby` | Agrupar filas por una columna |
| `size()` | Contar cuántas filas tiene cada grupo |
| Sesgo a la derecha | Cola larga hacia valores altos; media > mediana |
| Mediana | Valor central; poco sensible a extremos |
| Hipótesis | Explicación posible que aún no está confirmada |
