# Guía del Taller 01 — Primeros pasos con Scikit-learn, TensorFlow y PyTorch

> **Para qué sirve esta guía:** te acompaña paso a paso en el taller. Primero te explica la idea, luego te dice qué hacer y al final qué deberías observar. Los espacios `# TU CÓDIGO AQUÍ` los completas tú en tu notebook; las preguntas de reflexión las respondes con tus propias palabras.

## ¿Qué vas a lograr?

Al terminar el taller podrás:
- Entrenar un mismo modelo sencillo con los tres *frameworks* más usados de Python para IA y **comparar cómo se siente trabajar con cada uno**.
- Explicar con tus palabras en qué se diferencian (cantidad de código, control sobre el entrenamiento, tipo de problema para el que sirven).
- Relacionar lo que viviste con los seis criterios técnicos de la Semana 3: preprocesamiento, soporte GPU/rendimiento, facilidad de entrenamiento, interpretabilidad, comunidad/documentación e integración para despliegue.

## Antes de empezar

**Cómo trabajar**
- Usa un notebook de Jupyter (Colab o local) con `scikit-learn`, `tensorflow` y `torch` instalados. En Colab ya vienen instalados.
- Copia cada bloque de código en una celda, completa lo que dice `# TU CÓDIGO AQUÍ` y ejecútalo antes de seguir con el siguiente.
- Escribe las respuestas de reflexión en una celda de **texto (Markdown)**, no como comentarios dentro del código.
- Duración estimada: 60-90 minutos. Este taller retoma la actividad del anexo («probar el *framework* en 3 minutos»).

**El problema que se repite: `y = 2x`**
Casi todo el taller usa este problema: cuatro puntos `(1,2), (2,4), (3,6), (4,8)` donde la salida es siempre el doble de la entrada. Es tan simple que **sabes la respuesta de antemano** (para `X = 5` debería dar cerca de `10`). Eso es a propósito: así puedes concentrarte en comparar *cómo* trabaja cada *framework* y no en el problema.

**Vocabulario mínimo**

| Término | Qué significa |
|---|---|
| Modelo | Programa que aprende una relación entre datos de entrada (`X`) y salidas (`y`). |
| Entrenar (`fit`) | Dejar que el modelo ajuste sus parámetros usando los datos. |
| Predecir (`predict`) | Pedirle al modelo una salida para datos nuevos. |
| Entrenamiento / prueba | Datos con los que aprende / datos que no vio, para medir qué tan bien generaliza. |
| Error (MSE) | Promedio de los errores al cuadrado; cuanto más cerca de 0, mejor ajusta. |
| *Accuracy* | Proporción de aciertos de un clasificador (0.95 = acierta el 95 %). |
| Época | Una pasada completa del modelo por todos los datos de entrenamiento. |

---

## Paso 0 — Preparación: confirmar que los tres *frameworks* están disponibles

**La idea:** antes de programar, hay que comprobar que las herramientas están instaladas. Importar una librería que no existe produce un error `ModuleNotFoundError`, y imprimir su versión es la forma estándar de verificar que todo funciona (también te sirve para reportar problemas con precisión).

**Qué hacer:** ejecuta esta celda tal cual.

```python
import sklearn
import tensorflow as tf
import torch

print("scikit-learn:", sklearn.__version__)
print("tensorflow:", tf.__version__)
print("torch:", torch.__version__)
```

**Qué deberías ver:** tres líneas con un número de versión cada una. Puede que TensorFlow imprima además avisos en rojo o amarillo sobre CPU o GPU; son informativos, no errores.

**Pregunta de reflexión 0:** Si alguno de los tres imports falla, ¿qué comando usarías para instalarlo? (pista: `pip install ...`)

_Tu respuesta:_

---

## Parte 1 — Scikit-learn: machine learning clásico

**Contexto:** scikit-learn es la librería de referencia para el aprendizaje automático «clásico»: regresiones, vecinos más cercanos, árboles, etc. Su gran virtud es que todos los modelos se usan igual: **crear → `fit` → `predict`**. Elegir el algoritmo es tu decisión; el entrenamiento ya viene resuelto.

### Ejercicio 1.1 — Regresión lineal simple

**La idea:** una regresión lineal busca la recta `y = w·x + b` que mejor se ajusta a los puntos. Con nuestros datos, la recta perfecta es `w = 2` y `b = 0`, así que el error debería ser prácticamente cero.

**Qué hacer:** entrena un modelo de regresión lineal con estos datos y predice el valor para `X = 5`.

**Cosas que conviene saber**
- `X` es una **lista de listas** (`[[1], [2], ...]`): scikit-learn espera una *tabla* donde cada fila es una muestra y cada columna una variable. Por eso, aunque haya una sola variable, cada valor va entre corchetes.
- `predict` también recibe una tabla: para predecir con `X = 5` se escribe `[[5]]`, no `5`.
- `mean_squared_error(valores_reales, valores_predichos)` mide el error; usa los datos de entrenamiento para este ejercicio.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

# TU CÓDIGO AQUÍ
# 1. Crea el modelo (LinearRegression)
# 2. Entrénalo con fit(X, y)
# 3. Predice el valor para X=5
# 4. Calcula el error (mean_squared_error) sobre los datos de entrenamiento
```

**Qué deberías ver:** una predicción muy cercana a `10` y un error prácticamente `0` (algo como `1e-30` es cero para efectos prácticos: es ruido numérico del computador).

### Ejercicio 1.2 — Clasificación con datos reales (Iris)

**La idea:** `iris` es un conjunto clásico con 150 flores de tres especies, descritas por 4 medidas (largo y ancho del pétalo y del sépalo). Un **clasificador** aprende a decir a qué especie pertenece una flor a partir de sus medidas.

El algoritmo **KNN (vecinos más cercanos)** funciona así: para clasificar una flor nueva, busca las `k` flores de entrenamiento más parecidas (con `k = 3`, las 3 más cercanas) y se queda con la especie que más se repite entre ellas. Por eso es fácil de explicar: *«la clasificó así porque sus 3 vecinos eran de esa especie»*.

**Por qué se separa en entrenamiento y prueba:** si evaluaras al modelo con las mismas flores con las que aprendió, sería como poner un examen con las preguntas del taller. Se aparta una parte (`test_size=0.2` = 20 %) que el modelo **nunca ve** al entrenar, y solo con ella se mide el *accuracy*.

**Qué hacer:** completa los cuatro pasos del código.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

# TU CÓDIGO AQUÍ
# 1. Divide iris.data / iris.target en train/test (train_test_split, test_size=0.2)
# 2. Crea un KNeighborsClassifier(n_neighbors=3) y entrénalo
# 3. Predice sobre el conjunto de prueba
# 4. Calcula el accuracy con accuracy_score
```

**Pistas**
- `train_test_split` devuelve cuatro cosas, en este orden: `X_train, X_test, y_train, y_test`.
- `accuracy_score(valores_reales, valores_predichos)`.
- Para que tu resultado sea repetible, puedes añadir `random_state=42` al dividir.

**Qué deberías ver:** un *accuracy* alto, típicamente entre `0.93` y `1.0`. Con 30 flores de prueba, cada flor mal clasificada baja el resultado unos 3 puntos, así que no te sorprendas si cambia al variar la partición.

**Pregunta de reflexión 1:** ¿Qué tan fácil te resultó pasar de «tener los datos» a «tener un modelo entrenado» con scikit-learn? ¿Podrías explicar, en una frase, por qué el modelo tomó cada decisión (interpretabilidad)?

_Tu respuesta:_

---

## Parte 2 — TensorFlow/Keras: deep learning de alto nivel

**Contexto:** TensorFlow es una plataforma para **redes neuronales**, y Keras es su interfaz de alto nivel. Aquí **tú diseñas la arquitectura**: cuántas capas, cuántas neuronas, qué funciones de activación. El flujo es: **definir el modelo → `compile` (elegir optimizador y función de pérdida) → `fit` (entrenar)**.

### Ejercicio 2.1 — Red neuronal de regresión

**La idea:** una red con **una sola neurona** (`Dense(1)`) calcula exactamente `salida = w·x + b`, o sea, lo mismo que la regresión lineal. La diferencia es *cómo aprende*: no resuelve una fórmula, sino que empieza con valores aleatorios para `w` y `b` y los va corrigiendo poco a poco.

Piezas que vas a usar:
- **Optimizador (`sgd`):** la regla para corregir los parámetros en cada paso (descenso por gradiente).
- **Pérdida (`mse`):** el error que el optimizador intenta reducir.
- **Épocas (`epochs=200`):** cuántas veces recorre los 4 datos. Con pocas épocas el modelo aprende poco; por eso aquí hacen falta bastantes.

**Qué hacer:** completa los cuatro pasos.

```python
import numpy as np
import tensorflow as tf

X = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.0, 4.0, 6.0, 8.0])

# TU CÓDIGO AQUÍ
# 1. Crea un tf.keras.Sequential con una capa Input(shape=(1,)) y una Dense(1)
# 2. Compílalo con optimizer="sgd" y loss="mse"
# 3. Entrénalo con fit(X, y, epochs=200)
# 4. Predice el valor para X=5
```

> **Nota:** a diferencia de scikit-learn, Keras espera que los datos vengan como arrays de NumPy (`np.array(...)`), no como listas planas de Python. Si pasas una lista, verás un `ValueError`.

**Pistas**
- Añade `verbose=0` a `fit` si no quieres ver una línea por cada época.
- `modelo.predict(np.array([5.0]))` devuelve un arreglo, no un número suelto.

**Qué deberías ver:** una predicción cercana a `10`, pero **no exactamente 10** (por ejemplo `9.7`): la red aprende por aproximaciones y parte de valores aleatorios, por eso puede variar entre ejecuciones. Si quieres ver cómo mejora, prueba con más o menos épocas.

### Ejercicio 2.2 — Clasificación de imágenes pequeñas (dígitos escritos a mano)

**La idea:** `load_digits` trae 1 797 imágenes de dígitos (0 al 9) de 8×8 píxeles. Cada imagen se «aplana» en una fila de 64 números (la intensidad de cada píxel, de 0 a 16). Así, el problema es: dados 64 números, ¿qué dígito es?

Lo que debes entender de la arquitectura:

| Capa | Qué hace |
|---|---|
| `Input(shape=(64,))` | Declara que cada muestra trae 64 valores (los píxeles). |
| `Dense(32, activation="relu")` | 32 neuronas que combinan los píxeles. `relu` deja pasar los valores positivos y apaga los negativos; es lo que permite aprender relaciones que no son rectas. |
| `Dense(10, activation="softmax")` | 10 neuronas, una por dígito. `softmax` convierte sus salidas en probabilidades que suman 1; la mayor es la predicción. |

Y de la compilación:
- **`adam`**: un optimizador más sofisticado que `sgd`, que ajusta solo su velocidad de aprendizaje.
- **`sparse_categorical_crossentropy`**: la pérdida adecuada cuando las etiquetas son números enteros (0, 1, 2 …) y hay varias clases.
- **`metrics=["accuracy"]`**: para que `evaluate` te devuelva también el porcentaje de aciertos.

**Normalizar:** el código ya divide los píxeles entre 16 para que queden entre 0 y 1. Las redes entrenan mucho mejor con valores pequeños y parejos. Este es un ejemplo de **preprocesamiento**, uno de los seis criterios de la semana.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data / 16.0   # normalizamos los píxeles a [0, 1]
y = digits.target

# TU CÓDIGO AQUÍ
# 1. Divide X, y en train/test (test_size=0.2)
# 2. Crea un tf.keras.Sequential:
#    - Input(shape=(64,))
#    - Dense(32, activation="relu")
#    - Dense(10, activation="softmax")
# 3. Compílalo con optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# 4. Entrénalo con fit(X_train, y_train, epochs=20)
# 5. Evalúalo con evaluate(X_test, y_test)
```

**Qué deberías ver:** durante el entrenamiento, la pérdida (`loss`) baja y el `accuracy` sube época tras época. `evaluate` devuelve dos números: la pérdida y el *accuracy* de prueba, que suele quedar entre `0.93` y `0.98`.

**Pregunta de reflexión 2:** Compara cuántas líneas de código necesitaste para llegar a un modelo entrenado en el Ejercicio 1.2 (scikit-learn) frente al Ejercicio 2.2 (Keras). ¿En qué se nota que TensorFlow está pensado para redes neuronales y scikit-learn para modelos clásicos?

_Tu respuesta:_

---

## Parte 3 — PyTorch: control explícito del entrenamiento

**Contexto:** PyTorch es el otro gran *framework* de redes neuronales. Su filosofía es distinta: en lugar de una llamada `fit` que lo hace todo, **tú escribes el bucle de entrenamiento**. Es más trabajo, pero ves y controlas cada paso, y por eso es muy popular en investigación y para aprender cómo funcionan realmente las redes.

### Ejercicio 3.1 — Tensores básicos

**La idea:** un **tensor** es el «arreglo numérico» de PyTorch (como un `np.array`), con dos superpoderes: puede vivir en una GPU y puede calcular derivadas automáticamente (lo que hace posible entrenar). Dominar tensores es el primer paso.

**Qué hacer:** completa los cinco puntos.

```python
import torch

# TU CÓDIGO AQUÍ
# 1. Crea un tensor `a` con los valores [1.0, 2.0, 3.0]
# 2. Crea un tensor `b` de unos con torch.ones(3)
# 3. Súmalos e imprime el resultado
# 4. Imprime la forma (shape) de `a`
# 5. Imprime si hay GPU disponible con torch.cuda.is_available()
```

**Pistas:** `torch.tensor([...])` crea un tensor desde una lista; la suma se hace con `+` elemento a elemento; la forma se lee con `a.shape`.

**Qué deberías ver:** `tensor([2., 3., 4.])` y `torch.Size([3])`. Respecto a la GPU: **`False` es un resultado normal** en la mayoría de portátiles (y en Colab si no activaste la GPU); no significa que algo esté roto.

### Ejercicio 3.2 — Bucle de entrenamiento manual

**La idea:** entrenar una red consiste en repetir cuatro pasos. Esto es lo que Keras hace por dentro cuando llamas a `fit`:

| Paso | Instrucción | Qué ocurre |
|---|---|---|
| 1 | `opt.zero_grad()` | Borra los gradientes de la vuelta anterior (si no, se **acumulan** y el entrenamiento se descontrola). |
| 2 | `error = perdida(modelo(X), y)` | Calcula las predicciones y mide qué tan lejos están de la realidad. |
| 3 | `error.backward()` | **Retropropagación:** calcula cuánto contribuyó cada parámetro al error. |
| 4 | `opt.step()` | Corrige los parámetros un poquito en la dirección que reduce el error. |

Ya está definido el modelo `nn.Linear(1, 1)` (una neurona: `w·x + b`), el optimizador `SGD` con tasa de aprendizaje `lr=0.01` y la pérdida `MSELoss`. Tú escribes lo que va **dentro** del bucle, que se repite 200 veces.

```python
import torch
import torch.nn as nn

X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

modelo = nn.Linear(1, 1)
opt = torch.optim.SGD(modelo.parameters(), lr=0.01)
perdida = nn.MSELoss()

for _ in range(200):
    # TU CÓDIGO AQUÍ
    # 1. Reinicia los gradientes (opt.zero_grad())
    # 2. Calcula el error: perdida(modelo(X), y)
    # 3. Retropropaga el error (error.backward())
    # 4. Actualiza los pesos (opt.step())
    pass

print(modelo(torch.tensor([[5.0]])))
```

**Recuerda:** al completar el bucle puedes borrar la palabra `pass` (solo estaba para que el código fuera válido mientras estaba vacío).

**Qué deberías ver:** un tensor con un valor cercano a `10`, algo como `tensor([[9.8]], grad_fn=...)`. La parte `grad_fn` es normal: indica que PyTorch recuerda cómo calcular derivadas.

**Para experimentar (opcional):** comenta la línea de `opt.zero_grad()` y ejecuta de nuevo. Observa qué pasa con el resultado: es el error más común al empezar con PyTorch.

**Pregunta de reflexión 3:** En Keras, todo el entrenamiento ocurre dentro de una sola línea (`model.fit(...)`). En PyTorch tuviste que escribir cada paso (`zero_grad`, `backward`, `step`). ¿Qué ventaja y qué desventaja le ves a esta diferencia?

_Tu respuesta:_

---

## Parte 4 — Reto integrador (opcional)

**La idea:** acabas de resolver el **mismo problema** (`y = 2x`) con los tres *frameworks*: scikit-learn (Ejercicio 1.1), TensorFlow (Ejercicio 2.1) y PyTorch (Ejercicio 3.2). Ahora toca poner en palabras lo que viviste.

**Qué hacer:** completa la tabla con **tu experiencia**, no con lo que crees que «debería» decir. No hay una única respuesta correcta.

| Criterio | Scikit-learn | TensorFlow | PyTorch |
|---|---|---|---|
| Líneas de código aproximadas | | | |
| Facilidad de entrenamiento (1-5) | | | |
| Nivel de control sobre el entrenamiento (1-5) | | | |
| ¿Lo usarías para un modelo en producción? | | | |

**Cómo llenarla**
- *Líneas de código:* cuenta solo las que tú escribiste para llegar a la predicción.
- *Facilidad (1 = difícil, 5 = muy fácil)* y *Control (1 = poco, 5 = total):* piensa en cuánto tuviste que decidir y escribir tú.
- *Producción:* piensa en si te parecería sencillo guardar el modelo y ponerlo a funcionar en un servicio real (esto conecta con el criterio de integración para despliegue).

**Pregunta de cierre:** Retomando la pregunta orientadora del foro: según lo que hiciste en este taller, ¿qué criterio técnico (preprocesamiento, soporte GPU/rendimiento, facilidad de entrenamiento, interpretabilidad, comunidad/documentación o integración para despliegue) pesaría más para ti al elegir un *framework*, según el tipo de problema y los datos disponibles?

> **Para responder bien:** no basta con decir «prefiero X». Nombra un tipo de problema (por ejemplo, datos en tabla frente a imágenes), elige un criterio de los seis y explica por qué ese criterio es el que más importa en ese caso.

_Tu respuesta:_

---

## Lista de verificación final

- [ ] Ejecuté el Paso 0 y vi las tres versiones.
- [ ] Obtuve predicciones cercanas a `10` en los tres *frameworks*.
- [ ] Obtuve un *accuracy* razonable en `iris` y en `digits`.
- [ ] Respondí las preguntas de reflexión 0 a 3 en celdas de texto.
- [ ] (Opcional) Completé la tabla comparativa y la pregunta de cierre.

## Si algo falla

| Síntoma | Causa probable | Qué probar |
|---|---|---|
| `ModuleNotFoundError` | Falta instalar la librería. | `pip install scikit-learn`, `pip install tensorflow` o `pip install torch`. |
| `ValueError` en Keras con una lista | Keras no acepta listas de Python. | Conviértela con `np.array(...)`. |
| `Expected 2D array, got 1D array` en scikit-learn | `X` debe ser una tabla. | Usa `[[1], [2], ...]` o `.reshape(-1, 1)`. |
| El resultado de la red es `9.7` en vez de `10` | Es normal: las redes aprenden por aproximaciones. | Aumenta las épocas y verás que se acerca. |
| `torch.cuda.is_available()` da `False` | No hay GPU disponible. | Es un resultado válido; en Colab puedes activarla en *Entorno de ejecución*. |
| Aviso rojo de TensorFlow sobre CPU/GPU | Mensaje informativo. | Puedes ignorarlo si el código sigue ejecutándose. |
