# Guía del Taller 02 — Afianzando los seis criterios técnicos de la Semana 3

> **Para qué sirve esta guía:** en el Taller 01 aprendiste a *usar* los tres *frameworks*. Aquí aprenderás a **elegir entre ellos con evidencia**. Cada parte trabaja un criterio técnico: te explica la idea, te pide medirla tú mismo y te dice qué deberías observar. Los espacios `# TU CÓDIGO AQUÍ` los completas en tu notebook y las preguntas las respondes con tus propias palabras.

## ¿Qué vas a lograr?

Al terminar el taller podrás:
- Reconocer y corregir los errores de formato de datos más comunes en cada *framework*.
- Evitar la **fuga de datos** al escalar.
- Distinguir un modelo que se puede explicar de uno que no.
- Medir tiempos y comparar órdenes de magnitud en lugar de suponer.
- Guardar un modelo, recargarlo y comprobar que predice igual.
- Justificar la elección de un *framework* usando los seis criterios.

## Antes de empezar

**Cómo trabajar**
- Usa un notebook de Jupyter (Colab o local) con `scikit-learn`, `tensorflow` y `torch` instalados.
- Copia cada bloque en una celda, completa `# TU CÓDIGO AQUÍ` y ejecútalo antes de pasar al siguiente.
- Escribe las respuestas en una celda de **texto (Markdown)**, no como comentarios de código.
- Duración estimada: 60-90 minutos. Se recomienda haber hecho antes el Taller 01.

**Los seis criterios y dónde se trabajan**

| Criterio | Pregunta que responde | Parte del taller |
|---|---|---|
| Preprocesamiento | ¿Cómo debo preparar los datos para que el *framework* los acepte? | Parte 1 |
| Interpretabilidad | ¿Puedo explicar por qué el modelo decidió algo? | Parte 2 |
| Soporte GPU / rendimiento | ¿Cuánto tarda y en qué hardware corre? | Parte 3 |
| Facilidad de entrenamiento | ¿Cuánto código y cuántas decisiones exige? | Partes 3 y 4 |
| Integración para despliegue | ¿Cómo lo guardo y lo llevo a producción? | Parte 4 |
| Comunidad y documentación | ¿Es fácil encontrar ayuda y ejemplos? | Parte 5 (al justificar) |

---

## Parte 0 — Calentamiento: ¿qué *framework* es?

**La idea:** antes de medir nada, verifica que tienes claras las «huellas digitales» de cada *framework*. Si dudas, vuelve a mirar los ejercicios del Taller 01: la respuesta está en cómo se entrenó cada modelo allí.

**Qué hacer:** escribe en tu notebook (en una celda de texto) el nombre del *framework* (scikit-learn, TensorFlow/Keras o PyTorch) que corresponde a cada descripción.

| # | Descripción | Framework |
|---|---|---|
| a | Todo el entrenamiento ocurre en una llamada a `fit`; no hay que declarar capas. | |
| b | Hay que escribir `zero_grad()`, `backward()` y `step()` en un bucle. | |
| c | Se define el modelo con `Sequential`, se llama a `compile` y luego a `fit`. | |
| d | Los datos deben llegar como tensores `float32`. | |
| e | El modelo se guarda con `joblib.dump`. | |
| f | El modelo se guarda con `model.save("x.keras")`. | |

> **Pista:** hay dos descripciones de TensorFlow/Keras, dos de PyTorch y dos de scikit-learn. Si una de ellas te cuesta, ubica sus palabras clave (`fit` a secas, `compile`, `zero_grad`, `joblib`, `.keras`, `float32`).

---

## Parte 1 — Preprocesamiento: cada *framework* pide los datos a su manera

**Contexto:** «preprocesar» es dejar los datos en la forma que el modelo necesita. Cada *framework* tiene sus reglas, y cuando no se cumplen aparece un error. Aprender a **leer ese error** es una habilidad central: casi siempre el mensaje te dice qué falta.

| Framework | Qué espera de los datos |
|---|---|
| scikit-learn | Una tabla 2D (filas = muestras, columnas = variables). Acepta listas y `DataFrame`. |
| Keras (TensorFlow) | Arreglos de NumPy (`np.array`). **No** acepta listas de Python. |
| PyTorch | Tensores, del mismo tipo de dato que los pesos del modelo (`float32`). |

### Ejercicio 1.1 — Tres errores, tres arreglos

**La idea:** cada celda siguiente **está hecha para fallar**. No es un error tuyo: es el punto del ejercicio. Ejecútala, lee el mensaje de error y corrige **solo los datos** (no toques el modelo).

**Cómo abordar cada una**
1. Ejecuta la celda y copia la última línea del error.
2. Pregúntate: ¿qué forma o tipo de dato tengo, y cuál pide el *framework*?
3. Corrige, vuelve a ejecutar y comprueba que ya no falla.

**a) scikit-learn** espera una matriz 2D (filas = muestras, columnas = variables).
Mensaje que verás: `Expected 2D array, got 1D array instead`. Pista: cada valor de `X` debe ir dentro de su propia fila.

```python
from sklearn.linear_model import LinearRegression

X = [1, 2, 3, 4]          # <- 1D: provoca el error
y = [2, 4, 6, 8]

# TU CÓDIGO AQUÍ: corrige X y luego entrena y predice para X=5
modelo = LinearRegression().fit(X, y)
print(modelo.predict([[5]]))
```

**b) Keras** (Keras 3) no acepta listas de Python: exige arreglos de NumPy.
Mensaje que verás: `ValueError: Unrecognized data type ... (of type <class 'list'>)`. Pista: `np.array(...)` convierte una lista en arreglo de NumPy.

```python
import numpy as np
import tensorflow as tf

X = [1.0, 2.0, 3.0, 4.0]  # <- lista: provoca el error
y = [2.0, 4.0, 6.0, 8.0]

# TU CÓDIGO AQUÍ: convierte X e y a arreglos de NumPy

m = tf.keras.Sequential([tf.keras.layers.Input(shape=(1,)), tf.keras.layers.Dense(1)])
m.compile(optimizer="sgd", loss="mse")
m.fit(X, y, epochs=5, verbose=0)
print("Keras entrenó sin error")
```

**c) PyTorch** exige tensores y el mismo tipo de dato que los pesos (`float32`). Un arreglo de NumPy trae `float64` por defecto.
Mensaje que verás: un error de tipos (`... must be Tensor, not numpy.ndarray` o `mat1 and mat2 must have the same dtype`). Pista: `torch.tensor(X, dtype=torch.float32)`.

```python
import numpy as np
import torch
import torch.nn as nn

X = np.array([[1.0], [2.0], [3.0], [4.0]])   # <- float64 y ndarray: provoca el error
modelo = nn.Linear(1, 1)

# TU CÓDIGO AQUÍ: convierte X a un tensor float32

print(modelo(X).shape)
```

**Qué deberías ver:** en (a) una predicción cercana a `[10.]`; en (b) el texto «Keras entrenó sin error»; en (c) `torch.Size([4, 1])`.

### Ejercicio 1.2 — Escalar sin hacer trampa

**La idea:** muchos modelos funcionan mejor si las variables tienen una escala parecida. `StandardScaler` transforma cada variable para que tenga media 0 y desviación 1. Para hacerlo calcula esos valores con `fit` y los aplica con `transform`.

**Regla de oro:** el escalador se ajusta (`fit`) **solo con los datos de entrenamiento** y luego se aplica (`transform`) a los de entrenamiento y a los de prueba. En la práctica, los datos de prueba hacen de «datos futuros que aún no existen»: el modelo no puede haberlos visto, ni siquiera de forma indirecta.

**Qué hacer:** con `load_digits` (sin dividir entre 16 esta vez), estandariza los píxeles y entrena una regresión logística.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TU CÓDIGO AQUÍ
# 1. Crea un StandardScaler y ajústalo (fit) solo con X_train
# 2. Transforma X_train y X_test
# 3. Entrena LogisticRegression(max_iter=1000) con los datos escalados
# 4. Imprime el accuracy sobre la prueba
```

**Pistas:** el escalador se usa en dos tiempos, `escalador.fit(X_train)` y luego `escalador.transform(...)`; el *accuracy* lo da `modelo.score(X_test_escalado, y_test)`.

**Qué deberías ver:** un *accuracy* alrededor de `0.97`. Como comprobación opcional, la media de `X_train` escalado debería ser casi `0`.

**Pregunta de reflexión 1:** ¿Por qué es un error ajustar el `StandardScaler` con *todos* los datos antes de dividir en entrenamiento y prueba? ¿Qué parte de la prueba «se filtra» al modelo?

> **Pista para pensar:** ¿de dónde sale la media que usa el escalador? ¿Con qué datos se calculó? Esta situación se conoce como *fuga de datos* (*data leakage*).

_Tu respuesta:_

---

## Parte 2 — Interpretabilidad: ¿puedo explicar la decisión?

**Contexto:** un modelo **interpretable** es aquel cuyas decisiones puedes explicar a otra persona. Esto es obligatorio en áreas como crédito o salud, donde hay que justificar por qué se tomó una decisión. Hay modelos que se leen directamente (una recta, un árbol de reglas) y otros cuyos miles de parámetros no se pueden leer uno por uno (una red neuronal).

### Ejercicio 2.1 — Leer un modelo lineal

**La idea:** una regresión lineal aprende `y = w·x + b`. Esos dos números **son** el modelo: `coef_` guarda `w` (la pendiente) e `intercept_` guarda `b` (el valor cuando `x = 0`). Con ellos puedes escribir la fórmula completa a mano.

```python
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

# TU CÓDIGO AQUÍ
# 1. Entrena el modelo
# 2. Imprime coef_ (pendiente w) e intercept_ (sesgo b)
```

**Pista:** `coef_` es un arreglo; para un solo valor usa `modelo.coef_[0]`.

**Qué deberías ver:** `w` muy cercano a `2` y `b` muy cercano a `0`. ¿Coincide con la relación `y = 2x` que conoces? Así de transparente es un modelo interpretable.

### Ejercicio 2.2 — Un árbol de decisión sobre `iris`

**La idea:** un árbol de decisión clasifica haciendo preguntas encadenadas del tipo «¿el largo del pétalo es menor o igual a 2.45?». El resultado es una lista de reglas que cualquiera puede leer.
- `max_depth=3` limita la profundidad del árbol para que las reglas sean cortas y legibles.
- `export_text` imprime esas reglas.
- `feature_importances_` indica cuánto usó el árbol cada variable (suman 1). Una variable con importancia `0` no influyó en ninguna decisión.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

# TU CÓDIGO AQUÍ
# 1. Entrena el árbol (max_depth=3, random_state=42)
# 2. Imprime el accuracy sobre la prueba
# 3. Imprime las reglas: export_text(arbol, feature_names=iris.feature_names)
# 4. Imprime la importancia de cada variable (arbol.feature_importances_)
```

**Cómo leer las reglas:** cada línea con `|---` es una pregunta; la sangría indica a qué rama perteneces; `class: 0, 1, 2` es la especie de `iris` (setosa, versicolor, virginica). Recorre el árbol con una flor imaginaria para ver cómo llega a su clase.

**Qué deberías ver:** un *accuracy* alto (en esta partición, `1.0`), reglas que hablan sobre todo de **pétalos** y una importancia casi nula para los sépalos. Es una explicación que puedes defender ante un tercero.

### Ejercicio 2.3 — Contar parámetros de una red

**La idea:** un **parámetro** es un número que el modelo ajusta al aprender. Cuantos más hay, más difícil es entender qué hace cada uno. Vas a contar los de una red pequeña y compararlos con los de una regresión logística que resuelve el mismo problema (`digits`).

Cómo se cuentan en la red `64 → 32 → 10`: cada capa `Linear(entradas, salidas)` tiene `entradas × salidas` pesos más `salidas` sesgos. Puedes calcularlo a mano y luego comprobar con el código.

```python
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

red = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))

X, y = load_digits(return_X_y=True)
reg = LogisticRegression(max_iter=1000).fit(X / 16.0, y)

# TU CÓDIGO AQUÍ
# 1. Suma el número de elementos de red.parameters() (p.numel())
# 2. Suma reg.coef_.size + reg.intercept_.size
# 3. Imprime ambos totales
```

**Pista:** `sum(p.numel() for p in red.parameters())`.

**Qué deberías ver:** la red tiene unos 2 400 parámetros y la regresión logística unos 650. Verifica tu cálculo a mano: ¿coincide con lo que imprimió el código?

**Pregunta de reflexión 2:** Con el árbol de `iris` puedes decir «es versicolor porque el largo del pétalo es mayor que 2.45 y …». ¿Podrías dar una explicación igual de directa con la red de 2 410 parámetros? ¿En qué tipo de problema aceptarías perder esa explicación y en cuál no?

> **Pista para pensar:** compara un caso donde equivocarse tiene un costo bajo (recomendar una canción) con uno donde hay que justificar la decisión (negar un crédito).

_Tu respuesta:_

---

## Parte 3 — Rendimiento y facilidad de entrenamiento: mide, no supongas

**Contexto:** es común creer que «una red neuronal siempre es mejor». Aquí lo vas a comprobar con números. El **rendimiento** incluye cuánto tarda un entrenamiento y qué hardware usa; la **facilidad de entrenamiento** incluye cuánto código y cuántas decisiones exige cada *framework*.

### Ejercicio 3.1 — Cronometrar tres formas de entrenar `digits`

**La idea:** entrenarás tres modelos sobre el mismo problema (`digits`, 80 % para entrenamiento) y medirás tiempo y *accuracy* de cada uno:
1. **KNN** (scikit-learn): casi no «entrena»; se limita a guardar los datos.
2. **Red de Keras** (64-32-10, Adam, 20 épocas, `verbose=0`).
3. **Red de PyTorch** (misma arquitectura, bucle propio, 20 épocas, lotes de 32).

**Cómo medir el tiempo:**

```python
t0 = time.perf_counter()
# ... entrenar ...
segundos = time.perf_counter() - t0
```

**Cómo abordar la red de PyTorch (lotes de 32):** en cada época se baraja el orden de los datos y se recorre el conjunto en trozos de 32 muestras; para cada trozo se repiten los 4 pasos que ya conoces (`zero_grad`, pérdida, `backward`, `step`). Ten en cuenta que para clasificar con 10 clases la pérdida es `nn.CrossEntropyLoss()`, las etiquetas deben ser enteros largos (`dtype=torch.long`) y los datos `float32`. Esto último ya lo conoces de la Parte 1.

```python
import time
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

torch.manual_seed(0); tf.keras.utils.set_random_seed(0)

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype("float32")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

resultados = {}

# TU CÓDIGO AQUÍ
# Para cada modelo: t0 = time.perf_counter() ... entrenar ... guardar (segundos, accuracy)
# en resultados["nombre"]. Al final imprime una línea por modelo.
```

**Qué deberías ver:** una línea por modelo con su tiempo y su *accuracy*. Los segundos exactos **dependen de tu computador**; lo que importa es el **orden de magnitud**. Fíjate en cuál tarda milisegundos y cuáles segundos, y en cuál acierta más.

### Ejercicio 3.2 — ¿Hay GPU disponible?

**La idea:** una **GPU** acelera muchísimo el entrenamiento de redes grandes. No todos los equipos tienen una, ni todas las instalaciones la usan. Verás qué ve cada *framework* en tu equipo con `tf.config.list_physical_devices("GPU")` y `torch.cuda.is_available()`.

```python
import tensorflow as tf
import torch

# TU CÓDIGO AQUÍ
```

**Qué deberías ver:** si no tienes GPU configurada, `[]` y `False`. **Eso es normal y no es un error.** Ten presente que TensorFlow 2.11 o superior no usa la GPU en Windows nativo (se usa WSL2 o Colab), y que en PyTorch depende de qué versión instalaste (CPU o CUDA).

**Pregunta de reflexión 3:** Con tus tiempos del Ejercicio 3.1, ¿justifica la red neuronal su costo para `digits`? ¿Qué tendría que cambiar (datos, tamaño, hardware) para que sí valiera la pena?

> **Pista para pensar:** compara tiempo y *accuracy* de KNN contra los de las redes. ¿Cuánto ganó la red a cambio de cuánto tiempo extra? ¿Y si fueran un millón de imágenes a color?

_Tu respuesta:_

---

## Parte 4 — Despliegue: guardar, recargar y predecir igual

**Contexto:** un modelo entrenado solo sirve si puedes **guardarlo** y volver a usarlo después (en otro programa, en un servidor, mañana). «Desplegar» es justamente eso: llevarlo a donde alguien lo use. Cada *framework* lo hace a su manera:

| | scikit-learn | PyTorch |
|---|---|---|
| Guardar | `joblib.dump(modelo, "x.joblib")` | `torch.save(modelo.state_dict(), "x.pt")` |
| Recargar | `joblib.load("x.joblib")` | Crear de nuevo la arquitectura (`nn.Linear(1, 1)`) y cargar los pesos con `load_state_dict(...)` |
| Qué se guarda | El modelo completo | Solo los pesos |

### Ejercicio 4.1 — Un modelo que sobrevive a reiniciar el kernel

**La idea:** guardar el modelo, recargarlo en un objeto nuevo y comprobar que **predice exactamente lo mismo** que el original. Si la predicción cambia, algo salió mal en el guardado o la recarga.

**Cómo abordarlo:** el código de arriba entrena los dos modelos por ti (`sk` y `pt`). Tú guardas, recargas, comparas y mides el tamaño de los archivos. Para la comparación, calcula la predicción de `X = 5` con el modelo original y con el recargado, e imprime ambas.

```python
import os
import joblib
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

# --- scikit-learn
sk = LinearRegression().fit([[1], [2], [3], [4]], [2, 4, 6, 8])

# --- PyTorch
torch.manual_seed(0)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
pt = nn.Linear(1, 1)
opt = torch.optim.SGD(pt.parameters(), lr=0.01)
for _ in range(200):
    opt.zero_grad()
    nn.MSELoss()(pt(X), y).backward()
    opt.step()

# TU CÓDIGO AQUÍ
# 1. Guarda sk con joblib.dump(sk, "modelo_sk.joblib")
# 2. Guarda los pesos de pt con torch.save(pt.state_dict(), "modelo_pt.pt")
# 3. Recarga sk con joblib.load; para PyTorch crea un nn.Linear(1, 1) nuevo y usa
#    load_state_dict(torch.load("modelo_pt.pt", weights_only=True))
# 4. Compara la predicción para X=5 antes y después en cada uno
# 5. Imprime el tamaño de cada archivo con os.path.getsize(...)
```

**Qué deberías ver:** la predicción del modelo recargado **idéntica** a la del original en ambos casos, y archivos muy pequeños (de unos cientos de bytes a un par de kilobytes). Al terminar puedes borrar `modelo_sk.joblib` y `modelo_pt.pt`.

**Pregunta de reflexión 4:** ¿Qué diferencia práctica hay entre `joblib.dump(modelo, ...)` y `torch.save(modelo.state_dict(), ...)` al momento de recargar el modelo en otro programa?

> **Pista para pensar:** fíjate en lo que tuviste que escribir para PyTorch **antes** de poder cargar los pesos.

_Tu respuesta:_

---

## Parte 5 — Integrador: elegir con argumentos

**La idea:** no existe un *framework* «mejor» en abstracto; la elección depende del problema, los datos y el contexto. Ahora aplica los seis criterios (preprocesamiento, GPU/rendimiento, facilidad de entrenamiento, interpretabilidad, comunidad/documentación, despliegue) a casos concretos.

**Qué hacer:** para cada escenario, elige un *framework* de primera opción y justifica con **al menos un criterio**.

**Un buen método de cuatro preguntas para cada caso**
1. ¿Qué tipo de datos hay (tabla, imágenes, texto)? ¿Cuántos?
2. ¿Hay que explicar las decisiones a otras personas?
3. ¿Se necesita GPU o un gran volumen de cálculo?
4. ¿Dónde va a vivir el modelo (una API pequeña, un móvil, un navegador)?

| # | Escenario | Primera opción | Criterio y justificación |
|---|---|---|---|
| 1 | Un banco debe explicar por qué niega un crédito, con una tabla de 5 000 clientes. | | |
| 2 | Clasificar millones de fotos de productos y entrenar con GPU. | | |
| 3 | Investigar una arquitectura nueva y depurar cada paso del entrenamiento. | | |
| 4 | Publicar en una API pequeña un modelo de predicción de precios con datos tabulares. | | |
| 5 | Llevar un modelo a una aplicación móvil o al navegador. | | |

> **Cómo justificar bien:** una respuesta como «scikit-learn porque es fácil» es débil. Una buena respuesta nombra el criterio y lo conecta con el caso, por ejemplo: «*(framework)* porque el criterio de *(criterio)* es el que más pesa aquí, ya que *(razón del escenario)*». Apóyate en lo que mediste en las partes anteriores.

**Pregunta de cierre:** Con lo que mediste en este taller, completa la frase y justifícala en 3-4 líneas: *«Antes de usar una red neuronal, yo primero…»*

> **Pista para pensar:** recuerda los tiempos y el *accuracy* de la Parte 3, y qué tan explicables eran los modelos de la Parte 2.

_Tu respuesta:_

---

## Lista de verificación final

- [ ] Corregí los tres errores de formato de la Parte 1 y escalé sin fuga de datos.
- [ ] Leí `coef_` y las reglas del árbol, y conté los parámetros de la red.
- [ ] Medí tiempos y *accuracy* de los tres modelos y consulté la GPU.
- [ ] Guardé y recargué los modelos con la misma predicción.
- [ ] Completé la tabla de escenarios y respondí las preguntas de reflexión 1 a 4 y la de cierre en celdas de texto.

## Si algo falla

| Síntoma | Causa probable | Qué probar |
|---|---|---|
| `Expected 2D array, got 1D array` | `X` es una lista plana en scikit-learn. | Usa `[[1], [2], ...]` o `.reshape(-1, 1)`. |
| `ValueError: Unrecognized data type ... list` en Keras | Keras 3 no acepta listas. | Convierte con `np.array(...)`. |
| Error de `dtype` o de tipo en PyTorch | Datos `float64` o arreglo de NumPy. | `torch.tensor(X, dtype=torch.float32)`; etiquetas con `dtype=torch.long`. |
| `ConvergenceWarning` en `LogisticRegression` | No alcanzó a converger. | Verifica que usas `max_iter=1000`. |
| `tf.config.list_physical_devices("GPU")` devuelve `[]` | No hay GPU utilizable. | Es un resultado válido; no es un error de instalación. |
| Los tiempos cambian en cada ejecución | Dependen del equipo y de lo que esté corriendo. | Compara el orden de magnitud, no los segundos exactos. |
| Error al recargar los pesos de PyTorch | No creaste antes la arquitectura. | Crea `nn.Linear(1, 1)` y luego llama a `load_state_dict(...)`. |
