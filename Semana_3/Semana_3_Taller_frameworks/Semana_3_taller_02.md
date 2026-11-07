# Taller 02 — Afianzando los seis criterios técnicos de la Semana 3

**Instrucciones generales:**
- Trabaja en un notebook de Jupyter (Colab o local) con `scikit-learn`, `tensorflow` y `torch` instalados.
- Donde veas `# TU CÓDIGO AQUÍ`, reemplázalo por tu solución.
- Responde las preguntas en una celda de texto (Markdown), no como comentario de código.
- Duración estimada: 60-90 minutos.
- Este taller refuerza lo visto en el Taller 01 y en las diapositivas: **preprocesamiento, soporte GPU/rendimiento, facilidad de entrenamiento, interpretabilidad, comunidad/documentación e integración para despliegue**. Cada parte trabaja un criterio con evidencia medida por ti.

---

## Parte 0 — Calentamiento: ¿qué framework es?

Escribe el nombre del *framework* (scikit-learn, TensorFlow/Keras o PyTorch) que corresponde a cada descripción:

| # | Descripción | Framework |
|---|---|---|
| a | Todo el entrenamiento ocurre en una llamada a `fit`; no hay que declarar capas. | |
| b | Hay que escribir `zero_grad()`, `backward()` y `step()` en un bucle. | |
| c | Se define el modelo con `Sequential`, se llama a `compile` y luego a `fit`. | |
| d | Los datos deben llegar como tensores `float32`. | |
| e | El modelo se guarda con `joblib.dump`. | |
| f | El modelo se guarda con `model.save("x.keras")`. | |

---

## Parte 1 — Preprocesamiento: cada framework pide los datos a su manera

### Ejercicio 1.1 — Tres errores, tres arreglos

Cada celda siguiente **falla** porque los datos no tienen el formato que el *framework* espera. Lee el mensaje de error y corrige solo los datos (no el modelo).

**a) scikit-learn** espera una matriz 2D (filas = muestras, columnas = variables).


```python
from sklearn.linear_model import LinearRegression

X = [1, 2, 3, 4]          # <- 1D: provoca el error
y = [2, 4, 6, 8]

# TU CÓDIGO AQUÍ: corrige X y luego entrena y predice para X=5
modelo = LinearRegression().fit(X, y)
print(modelo.predict([[5]]))
```

**b) Keras** (Keras 3) no acepta listas de Python: exige arreglos de NumPy.


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


```python
import numpy as np
import torch
import torch.nn as nn

X = np.array([[1.0], [2.0], [3.0], [4.0]])   # <- float64 y ndarray: provoca el error
modelo = nn.Linear(1, 1)

# TU CÓDIGO AQUÍ: convierte X a un tensor float32

print(modelo(X).shape)
```

### Ejercicio 1.2 — Escalar sin hacer trampa

Con `load_digits` (sin dividir entre 16 esta vez), estandariza los píxeles con `StandardScaler`. **Regla de oro:** el escalador se ajusta (`fit`) solo con los datos de entrenamiento y luego se aplica (`transform`) a los de prueba.


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

**Pregunta de reflexión 1:** ¿Por qué es un error ajustar el `StandardScaler` con *todos* los datos antes de dividir en entrenamiento y prueba? ¿Qué parte de la prueba "se filtra" al modelo?

_Tu respuesta:_ 

---

## Parte 2 — Interpretabilidad: ¿puedo explicar la decisión?

### Ejercicio 2.1 — Leer un modelo lineal

Entrena `LinearRegression` con `y = 2x` y lee sus parámetros con `coef_` e `intercept_`.


```python
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4]]
y = [2, 4, 6, 8]

# TU CÓDIGO AQUÍ
# 1. Entrena el modelo
# 2. Imprime coef_ (pendiente w) e intercept_ (sesgo b)
```

### Ejercicio 2.2 — Un árbol de decisión sobre `iris`

Entrena un `DecisionTreeClassifier(max_depth=3, random_state=42)` con `iris`, imprime sus reglas con `export_text` y las importancias de cada variable.


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

### Ejercicio 2.3 — Contar parámetros de una red

Cuenta los parámetros de una red pequeña de PyTorch (`64 → 32 → 10`) y compáralos con los de la regresión logística de scikit-learn sobre `digits` (que tiene `coef_` e `intercept_`).


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

**Pregunta de reflexión 2:** Con el árbol de `iris` puedes decir "es versicolor porque el largo del pétalo es mayor que 2.45 y …". ¿Podrías dar una explicación igual de directa con la red de 2 410 parámetros? ¿En qué tipo de problema aceptarías perder esa explicación y en cuál no?

_Tu respuesta:_ 

---

## Parte 3 — Rendimiento y facilidad de entrenamiento: mide, no supongas

### Ejercicio 3.1 — Cronometrar tres formas de entrenar `digits`

Mide con `time.perf_counter()` cuánto tarda cada entrenamiento sobre `digits` (80 % entrenamiento) y su accuracy: **KNN** (scikit-learn), **red de Keras** (64-32-10, Adam, 20 épocas, `verbose=0`) y **red de PyTorch** (misma arquitectura, bucle propio, 20 épocas, lotes de 32).


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

resultados = {}

# TU CÓDIGO AQUÍ
# Para cada modelo: t0 = time.perf_counter() ... entrenar ... guardar (segundos, accuracy)
# en resultados["nombre"]. Al final imprime una línea por modelo.
```

### Ejercicio 3.2 — ¿Hay GPU disponible?

Consulta los dispositivos que ve cada *framework*: `tf.config.list_physical_devices("GPU")` y `torch.cuda.is_available()`.


```python
import tensorflow as tf
import torch

# TU CÓDIGO AQUÍ
```

**Pregunta de reflexión 3:** Con tus tiempos del Ejercicio 3.1, ¿justifica la red neuronal su costo para `digits`? ¿Qué tendría que cambiar (datos, tamaño, hardware) para que sí valiera la pena?

_Tu respuesta:_ 

---

## Parte 4 — Despliegue: guardar, recargar y predecir igual

### Ejercicio 4.1 — Un modelo que sobrevive a reiniciar el kernel

Entrena el modelo `y = 2x` en scikit-learn y en PyTorch, guárdalos en disco, recárgalos y comprueba que la predicción para `X = 5` es **idéntica** antes y después.


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

**Pregunta de reflexión 4:** ¿Qué diferencia práctica hay entre `joblib.dump(modelo, ...)` y `torch.save(modelo.state_dict(), ...)` al momento de recargar el modelo en otro programa?

_Tu respuesta:_ 

---

## Parte 5 — Integrador: elegir con argumentos

Para cada escenario, elige un *framework* de primera opción y justifica con **al menos un criterio** de los seis (preprocesamiento, GPU/rendimiento, facilidad de entrenamiento, interpretabilidad, comunidad/documentación, despliegue).

| # | Escenario | Primera opción | Criterio y justificación |
|---|---|---|---|
| 1 | Un banco debe explicar por qué niega un crédito, con una tabla de 5 000 clientes. | | |
| 2 | Clasificar millones de fotos de productos y entrenar con GPU. | | |
| 3 | Investigar una arquitectura nueva y depurar cada paso del entrenamiento. | | |
| 4 | Publicar en una API pequeña un modelo de predicción de precios con datos tabulares. | | |
| 5 | Llevar un modelo a una aplicación móvil o al navegador. | | |

**Pregunta de cierre:** Con lo que mediste en este taller, completa la frase y justifícala en 3-4 líneas: *"Antes de usar una red neuronal, yo primero…"*

_Tu respuesta:_ 
