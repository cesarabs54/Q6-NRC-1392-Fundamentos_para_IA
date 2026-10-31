# Semana 1 — Cuadro técnico consolidado

**Tema:** Introducción a Python para IA: intérprete (REPL), script y notebook; tipos de datos y variables; operadores; estructuras de datos (listas, tuplas, conjuntos, diccionarios); control de flujo; funciones; manejo de errores y archivos; módulos.
**Resultados de aprendizaje:** RA1 «Aplica lenguaje de programación Python» y RA2 «Experimenta con el intérprete de Python».
**Actividad de la semana:** no hay actividad calificada con rúbrica en el repositorio. Hay dos entregables prácticos: (a) la **actividad en Colab** de la guía de la sesión (3 tareas: variables y tipos, condicional de nota, función promedio) y (b) los **12 ejercicios de clase** (`semana_1_Ejercicios_básicos.md`, con clave en `semana_1_profesor.md`), todos sobre un mismo escenario: un registro de experimentos de *Machine Learning* en Python puro, sin librerías.
**Material de la semana:**

| Archivo | Contenido |
|---|---|
| `Semana_1_Introduccion_Python_para_IA.pdf` | 8 diapositivas: bienvenida, popularidad de Python, objetivo del curso, resultados de aprendizaje y conclusiones |
| `Semana_1_Introduccion_Python.pdf` | Guía de la sesión (13 páginas, «Julio 2026»): por qué Python, REPL / script / notebook, Colab, tipos, `if` / `for` / `while`, funciones y la actividad en Colab |
| `semana_1_Ejercicios_básicos.md` | 12 ejercicios con código de partida (`____`, `# TODO`) y salida esperada |
| `semana_1_profesor.md` | Los mismos 12 ejercicios con solución y notas para el docente |

> Todas las salidas de este documento se recalcularon ejecutando el código (Python 3.11.9, Windows). Las marcadas con ✅ coinciden con la clave del profesor (`semana_1_profesor.md`) o con la guía de la sesión; las marcadas con ➕ son ampliaciones que se pueden aplicar y que **no** están en esos materiales. Las cifras de popularidad de las diapositivas (+50 %, 31,73 %, IEEE 2021) no se verificaron: se citan tal como aparecen.
>
> Esta semana no hay datos tabulares ni estadística (pandas llega en la Semana 2), así que, como en la Semana 2, la sección 3 reemplaza el «cuadro de pruebas» por un cuadro de errores y excepciones.

---

## 1. Procedimientos de la semana (de punta a punta)

| Etapa | Concepto / sintaxis | Para qué sirve | Ejemplo en los ejercicios |
|---|---|---|---|
| Intérprete | `python`, `>>>`, `type()`, `dir()`, `help()`, `exit()` | Probar ideas línea a línea; el REPL muestra el valor de cada expresión sin `print` | Ej. 1: `3 + 4 * 2` → `11`; `type(3.5)` → `float`; `"IA" * 3` → `'IAIAIA'` |
| Variables y tipos | `int`, `float`, `str`, `bool`, `None`; asignación sin declarar tipo | Representar un dato con el tipo correcto | Ej. 2: `"RandomForest"` `str`, `0.87` `float`, `20` `int`, `True` `bool`, `None` `NoneType` |
| Operadores | Aritméticos, de comparación, f-string con `:.2f` | Calcular métricas derivadas y comparar | Ej. 3: error A 0.13, error B 0.09, diferencia 0.04, `b_es_mejor` `True` |
| Listas | `append`, `remove`, `sort(reverse=True)`, `sorted`, `len`, índice | Colección ordenada y mutable | Ej. 4: `[0.95, 0.91, 0.87, 0.82, 0.75]`, mejor 0.95, total 5 |
| Tuplas | `(a, b, c)`, inmutables | Datos que ya no deben cambiar | Ej. 5: `(0.01, 20, 'adam')`; asignar `[0]` lanza `TypeError` |
| Conjuntos | `set(lista)`, `len` | Eliminar repetidos, contar distintos | Ej. 5: 5 nombres con repetidos → 3 modelos distintos |
| Diccionarios | `{llave: valor}`, `d[llave] = v`, `keys()`, `get()` | Agrupar campos relacionados de un registro | Ej. 6: `dict_keys(['modelo', 'accuracy', 'epocas', 'convergio', 'dataset_size'])` |
| Condicionales | `if` / `elif` / `else`, `match` / `case` (Python 3.10+) | Decidir según una condición | Ej. 7: 0.95 → Excelente, 0.80 → Aceptable, 0.60 → Insuficiente; `"sgd"` → *Stochastic Gradient Descent* |
| Bucles | `for x in lista`, `while cond`, acumulador | Recorrer y acumular; repetir mientras se cumpla algo | Ej. 8: promedio 0.853, mejor modelo SVM (0.91), cuenta regresiva 3 → 1 |
| Funciones | `def`, `return`, `lambda`, `sorted(key=...)`, comprehension | Encapsular lógica repetida y reutilizar el resultado | Ej. 9: `clasificar_experimento(0.91)` → `Excelente`; orden `['SVM', 'RandomForest', 'KNN']` |
| Errores | `try` / `except`, `raise` | Que un dato mal capturado no rompa el programa | Ej. 10: `"0.87"` → 0.87; `"abc"` y `"1.5"` → mensaje y `None` |
| Archivos | `with open(..., "w"/"r")`, `write`, `strip`, `split(",")` | Guardar y recuperar datos entre ejecuciones | Ej. 11: `historial.txt` con una línea `modelo,accuracy` por experimento |
| Módulos | `from archivo import función` | Separar la lógica del punto de entrada; mismo mecanismo que `import numpy` | Ej. 12: `experimentos.py` + `main.py` → `Promedio: 0.853`, `Mejor modelo (SVM): Excelente` |
| Entorno | Google Colab: celda + `Shift + Enter` | Notebook en la nube, sin instalar nada | Actividad en Colab (sección 4) |

---

## 2. Qué estructura de datos usar

| Necesito… | Estructura | Ejemplo | Cuidado | Uso |
|---|---|---|---|---|
| Una secuencia que crece, se ordena o se filtra | Lista `[]` | `accuracies = [0.75, 0.82, ...]` | `sort()` modifica en sitio y **devuelve `None`**: `a = a.sort()` deja `a = None`. `sorted()` devuelve una lista nueva. `remove()` borra por **valor** (el primero que encuentre) | ✅ |
| Valores que no deben cambiar | Tupla `()` | `(0.01, 20, "adam")` | `t[0] = 0.1` → `TypeError: 'tuple' object does not support item assignment` | ✅ |
| Quitar repetidos / contar distintos | Conjunto `set()` | 5 nombres → 3 | No garantiza orden: el `print` puede salir distinto al de la lista | ✅ |
| Campos con nombre de un mismo registro | Diccionario `{}` | `{"modelo": "SVM", "accuracy": 0.91}` | `d["x"]` con llave inexistente → `KeyError`; `d.get("x")` devuelve `None` y `d.get("x", 0)` devuelve `0` | ✅ (acceso) / ➕ (`get`) |
| Varios registros del mismo tipo | Lista de diccionarios | `experimentos` (Ej. 8, 9, 12) | Es la forma que luego toman las filas de un `DataFrame` (Semana 2) | ✅ |
| Ausencia de valor | `None` | `observaciones = None` | No es `0`, ni `""`, ni `False`. Se compara con `is None`. Usar `"N/A"` lo vuelve un texto más | ✅ |

Guía rápida para elegir:

| Si quiero… | Usar |
|---|---|
| Mantener el orden y poder cambiar elementos | Lista |
| Proteger los valores de cambios accidentales | Tupla |
| Saber cuántos valores distintos hay | `len(set(...))` |
| Buscar por nombre en vez de por posición | Diccionario |
| Transformar o filtrar una lista en una línea | Comprehension: `[e["modelo"] for e in experimentos]` |
| Obtener el máximo por un criterio | `max(lista, key=lambda e: e["accuracy"])` |

---

## 3. Errores y excepciones que aparecen

| Situación | Excepción | Mensaje (Python 3.11.9) | Cómo evitarlo | Uso |
|---|---|---|---|---|
| `float("abc")` | `ValueError` | `could not convert string to float: 'abc'` | `try` / `except ValueError` | ✅ (Ej. 10) |
| Valor fuera de [0, 1] | `ValueError` propio | `Accuracy fuera de rango` | `raise ValueError("...")` | ✅ (Ej. 10) |
| `t[0] = 0.1` en una tupla | `TypeError` | `'tuple' object does not support item assignment` | Usar lista si el dato debe cambiar | ✅ (Ej. 5) |
| `float(None)` | `TypeError` | `float() argument must be a string or a real number, not 'NoneType'` | **No** lo atrapa `except ValueError` (salvedad 6) | ➕ |
| `float("4,5")` (coma decimal) | `ValueError` | `could not convert string to float: '4,5'` | `.replace(",", ".")` antes de convertir | ➕ |
| `int("3.5")` | `ValueError` | `invalid literal for int() with base 10: '3.5'` | `int(float("3.5"))` si se acepta truncar | ➕ |
| `d["nada"]` | `KeyError` | `'nada'` | `d.get("nada")` | ➕ |
| `sum([]) / len([])` | `ZeroDivisionError` | `division by zero` | Preguntar `if lista:` antes de promediar | ➕ |
| `max([])` | `ValueError` | `max() arg is an empty sequence` | Igual que arriba, o `max(lista, default=None)` | ➕ |
| `open("noexiste.txt")` | `FileNotFoundError` | `[Errno 2] No such file or directory: 'noexiste.txt'` | Revisar la carpeta de trabajo; usar modo `"w"` para crear | ➕ |
| `float("nan")` | — (no lanza) | `0 <= nan <= 1` es `False` | La validación del Ej. 10 lo rechaza, pero con el mensaje «fuera de rango» | ➕ |

Regla práctica: capturar la excepción **más específica** que se espera. Un `except ValueError` no atrapa un `TypeError` ni un `ZeroDivisionError`; y `raise Exception(...)` en vez de `ValueError` tampoco sería atrapado por ese `except` (así lo advierte la nota del profesor del Ej. 10).

---

## 4. Actividad en Colab (guía de la sesión)

| Tarea | Enunciado | Solución de referencia | Salida verificada | Uso |
|---|---|---|---|---|
| 1. Variables y tipos | Nombre, edad y «estudiante activo»; imprimir el tipo de cada una | `nombre = "Ana"`, `edad = 20`, `activo = True` y `type()` de cada una | `str`, `int`, `bool` | ➕ (la guía no trae solución) |
| 2. Condicional | Recibir una nota entre 0 y 5 e imprimir «Aprobó» o «Reprobó» | `if nota >= 3.0` (umbral del ejemplo de la guía) | 4.2 → Aprobó · 3.0 → Aprobó · 2.9 → Reprobó | ✅ (patrón de la guía) / ➕ (casos) |
| 3. Función | Dos números → promedio; probar con al menos dos pares | `def promedio(a, b): return (a + b) / 2` | `promedio(4.0, 3.5)` = 3.75 ✅ · `(3, 5)` = 4.0 · `(0, 5)` = 2.5 | ✅ / ➕ |

Condiciones de calidad que se pueden cuidar (la guía no fija rúbrica): que el notebook ejecute de principio a fin sin error, que cada tarea tenga su celda, y que se usen `return` (no solo `print`) en la función.

---

## 5. Ejercicios de clase — resultado verificado

| Ej. | Tema | Resultado recalculado | Observación |
|---|---|---|---|
| 1 | Intérprete | `11`, `<class 'float'>`, `'IAIAIA'`, `'RandomForest'`; `dir(...)[-5:]` → `['swapcase', 'title', 'translate', 'upper', 'zfill']` ✅ | La clave imprime `'upper'` dos veces (salvedad 4). El REPL muestra el valor sin `print`; un script no |
| 2 | Variables y tipos | `str`, `float`, `int`, `bool`, `NoneType` ✅ | `None` ≠ `0`, `""`, `False`: las cuatro comparaciones con `None` dan `False` |
| 3 | Operadores | `Error A: 0.13`, `Error B: 0.09`, `Diferencia: 0.04`, `True` ✅ | Internamente `1 - 0.91` = 0.08999999999999997 y `0.91 - 0.87` = 0.040000000000000036; el formato `:.2f` lo oculta (salvedad 5) |
| 4 | Listas | `[0.95, 0.91, 0.87, 0.82, 0.75]`, mejor 0.95, total 5 ✅ | `[3, 1, 2].sort()` devuelve `None` (verificado) |
| 5 | Tuplas y conjuntos | `(0.01, 20, 'adam')`; 3 modelos distintos ✅ | `TypeError: 'tuple' object does not support item assignment`; el orden del conjunto varía |
| 6 | Diccionarios | `Accuracy: 0.87`, `dict_keys([... 'dataset_size'])` ✅ | `keys()` devuelve una vista, no una lista: se imprime como `dict_keys([...])` |
| 7 | Condicionales | Excelente / Aceptable / Insuficiente / *Stochastic Gradient Descent* ✅ | 0.9 cae en «Excelente» (`>=`); en el `elif` no hace falta repetir `< 0.9`. `match` exige Python 3.10+ |
| 8 | Bucles | Suma 2.56, promedio 0.853, mejor SVM (0.91), cuenta 3, 2, 1 ✅ | Sin `epocas_restantes -= 1` el `while` no termina (se interrumpe con `Ctrl+C`) |
| 9 | Funciones | `Excelente`, `Promedio: 0.853`, `['SVM', 'RandomForest', 'KNN']` ✅ | `return` permite usar el resultado dentro de un f-string; `print` no |
| 10 | Errores | `Resultado: 0.87` · `Dato inválido (abc): could not convert string to float: 'abc'` · `Dato inválido (1.5): Accuracy fuera de rango` ✅ | El mismo `except ValueError` atrapa el error automático y el `raise` propio |
| 11 | Archivos | `Modelo: RandomForest, Accuracy: 0.87` / `Modelo: SVM, Accuracy: 0.91` ✅ | Cada línea queda `'RandomForest,0.87\n'`; lo leído es `str`, no `float` |
| 12 | Módulos | `Promedio: 0.853` / `Mejor modelo (SVM): Excelente` ✅ | Ejecutado desde la misma carpeta y también desde otra (`python ../main.py`): funciona porque la carpeta del script entra en `sys.path` |

---

## 6. Preguntas de reflexión y evidencia

| Pregunta | Respuesta breve | Evidencia (ejercicio) |
|---|---|---|
| ¿Qué diferencia hay entre el REPL y un `.py`? | El REPL evalúa cada expresión y muestra el resultado; el script solo imprime lo que se le pide y no conserva estado entre ejecuciones | Ej. 1 ✅ |
| ¿Para qué sirve `dir()` y `help()`? | `dir(obj)` lista atributos y métodos; `help(str.upper)` muestra la documentación (se sale con `q`) | Ej. 1 ✅ |
| ¿Por qué `None` y no `"N/A"`? | `None` es un tipo propio; `"N/A"` es un texto más y rompe `if x is None` | Ej. 2 ✅ |
| ¿`sort()` o `sorted()`? | `sort()` cambia la lista y devuelve `None`; `sorted()` devuelve otra | Ej. 4 ✅ |
| ¿Por qué una tupla para hiperparámetros? | Porque un experimento cerrado no debe cambiar | Ej. 5 ✅ |
| ¿Por qué un diccionario y no variables sueltas? | Agrupa campos relacionados y anticipa las filas de un `DataFrame` | Ej. 6 ✅ |
| ¿`return` o `print`? | Solo `return` permite reutilizar el resultado | Ej. 9 ✅ |
| ¿Por qué `with open(...)`? | Cierra el archivo aunque ocurra una excepción | Ej. 11 ✅ |
| ¿Qué tiene que ver `main.py` con `import numpy`? | Es el mismo mecanismo; solo cambia quién escribió el módulo | Ej. 12 ✅ |

---

## 7. Código listo para pegar

Los bloques se ejecutaron sin error con Python 3.11.9.

### 7.1 Ejercicios 2 a 10 en un solo script (versión del profesor)

```python
# Ej. 2-3: tipos y operadores
nombre_modelo, accuracy, epocas, convergio, observaciones = "RandomForest", 0.87, 20, True, None
for v in (nombre_modelo, accuracy, epocas, convergio, observaciones):
    print(v, type(v))

acc_a, acc_b = 0.87, 0.91
print(f"Error A: {1 - acc_a:.2f}", f"Error B: {1 - acc_b:.2f}",
      f"Diferencia: {acc_b - acc_a:.2f}", acc_b > acc_a)

# Ej. 4-6: listas, tuplas, conjuntos, diccionarios
accuracies = [0.75, 0.82, 0.91, 0.68, 0.87]
accuracies.append(0.95)
accuracies.remove(0.68)
accuracies.sort(reverse=True)
print(accuracies, accuracies[0], len(accuracies))

hiperparametros = (0.01, 20, "adam")
modelos = set(["RandomForest", "SVM", "RandomForest", "KNN", "SVM"])
print(hiperparametros, len(modelos))

experimento = {"modelo": "RandomForest", "accuracy": 0.87, "epocas": 20, "convergio": True}
experimento["dataset_size"] = 5000
print(experimento["accuracy"], experimento.keys())

# Ej. 7-9: condicionales, bucles, funciones
def clasificar_experimento(accuracy):
    if accuracy >= 0.9:
        return "Excelente"
    elif accuracy >= 0.75:
        return "Aceptable"
    else:
        return "Insuficiente"

def promedio_accuracy(lista_experimentos):
    return sum(e["accuracy"] for e in lista_experimentos) / len(lista_experimentos)

experimentos = [
    {"modelo": "RandomForest", "accuracy": 0.87},
    {"modelo": "SVM", "accuracy": 0.91},
    {"modelo": "KNN", "accuracy": 0.78},
]
print([clasificar_experimento(a) for a in (0.95, 0.80, 0.60)])
print(f"Promedio: {promedio_accuracy(experimentos):.3f}")
print([e["modelo"] for e in sorted(experimentos, key=lambda e: e["accuracy"], reverse=True)])

epocas_restantes = 3
while epocas_restantes > 0:
    print("Épocas restantes:", epocas_restantes)
    epocas_restantes -= 1

# Ej. 10: validación
def validar_accuracy(valor):
    try:
        numero = float(valor)
        if not (0 <= numero <= 1):
            raise ValueError("Accuracy fuera de rango")
        return numero
    except ValueError as e:
        print(f"Dato inválido ({valor}): {e}")
        return None

for valor in ["0.87", "abc", "1.5"]:
    print("Resultado:", validar_accuracy(valor))
```

### 7.2 Versión ampliada ➕: tolerancia en flotantes, validación robusta y archivo con codificación

```python
import math

experimentos = [
    {"modelo": "RandomForest", "accuracy": 0.87, "epocas": 20, "convergio": True},
    {"modelo": "SVM", "accuracy": 0.91, "epocas": 15, "convergio": True},
    {"modelo": "KNN", "accuracy": 0.78, "epocas": 1, "convergio": False},
]

def clasificar_experimento(accuracy):
    if accuracy >= 0.9:
        return "Excelente"
    elif accuracy >= 0.75:
        return "Aceptable"
    return "Insuficiente"

def promedio_accuracy(lista):
    if not lista:                         # evita ZeroDivisionError
        return None
    return sum(e["accuracy"] for e in lista) / len(lista)

def validar_accuracy(valor):
    try:
        numero = float(str(valor).replace(",", "."))   # acepta "0,87"; None queda como 'None' -> ValueError
    except (ValueError, TypeError) as e:
        print(f"Dato inválido ({valor!r}): {e}")
        return None
    if not (0 <= numero <= 1):                         # también rechaza nan
        print(f"Dato inválido ({valor!r}): fuera de rango")
        return None
    return numero

# 1) Flotantes: comparar con tolerancia, no con ==
print(0.91 - 0.87 == 0.04, math.isclose(0.91 - 0.87, 0.04), round(0.91 - 0.87, 2))

# 2) Validación robusta
print([validar_accuracy(v) for v in ["0.87", "0,87", "abc", "1.5", None, "nan"]])

# 3) Estadísticas con comprehensions
convergidos = [e for e in experimentos if e["convergio"]]
print(len(convergidos), round(promedio_accuracy(convergidos), 3), promedio_accuracy([]))
print({e["modelo"]: clasificar_experimento(e["accuracy"]) for e in experimentos})
print(max(experimentos, key=lambda e: e["accuracy"])["modelo"],
      min(experimentos, key=lambda e: e["accuracy"])["modelo"])

# 4) Persistencia con codificación explícita y lectura de vuelta a float
with open("historial.txt", "w", encoding="utf-8") as f:
    for e in experimentos:
        f.write(f"{e['modelo']},{e['accuracy']}\n")
with open("historial.txt", encoding="utf-8") as f:
    leidos = [(m, float(a)) for m, a in (l.strip().split(",") for l in f)]
print(leidos)
```

Salida: `False True 0.04` · `[0.87, 0.87, None, None, None, None]` (con un mensaje por cada dato inválido) · `2 0.89 None` · `{'RandomForest': 'Aceptable', 'SVM': 'Excelente', 'KNN': 'Aceptable'}` · `SVM KNN` · `[('RandomForest', 0.87), ('SVM', 0.91), ('KNN', 0.78)]`.

### 7.3 Ejercicio 12: dos archivos (`experimentos.py` y `main.py`)

```python
# experimentos.py
def clasificar_experimento(accuracy):
    if accuracy >= 0.9:
        return "Excelente"
    elif accuracy >= 0.75:
        return "Aceptable"
    else:
        return "Insuficiente"

def promedio_accuracy(lista_experimentos):
    suma = sum(exp["accuracy"] for exp in lista_experimentos)
    return suma / len(lista_experimentos)
```

```python
# main.py
from experimentos import clasificar_experimento, promedio_accuracy

experimentos = [
    {"modelo": "RandomForest", "accuracy": 0.87},
    {"modelo": "SVM", "accuracy": 0.91},
    {"modelo": "KNN", "accuracy": 0.78},
]
promedio = promedio_accuracy(experimentos)
mejor = max(experimentos, key=lambda e: e["accuracy"])
print(f"Promedio: {promedio:.3f}")
print(f"Mejor modelo ({mejor['modelo']}): {clasificar_experimento(mejor['accuracy'])}")
```

---

## 8. Salvedades que conviene conocer

**Guía de la sesión y diapositivas**

1. **Cobertura desigual.** La guía de la sesión solo desarrolla variables y tipos, `if` / `for` / `while`, funciones y el REPL. Listas, tuplas, conjuntos, diccionarios, errores, archivos, módulos, `match` y `lambda` aparecen **solo** en los 12 ejercicios. En el texto extraído del PDF de 8 diapositivas, el contenido anuncia «Librerías de Python aplicadas a la IA» (punto 5), pero no encontré una diapositiva que lo desarrolle.
2. **Diapositivas distintas.** Los dos PDF no son versiones del mismo documento: uno es de apertura (bienvenida, popularidad, conclusiones) y el otro es la guía técnica. Las cifras de popularidad no están respaldadas con fuente verificable en el repositorio.
3. **Sin rúbrica.** El repositorio no trae puntaje ni criterios para la actividad en Colab ni para los 12 ejercicios. Antes de entregar, conviene confirmar en la plataforma el formato y la fecha.

**Ejercicios**

4. **Ej. 1: la salida de `dir()` en la clave está mal.** Muestra `['title', 'translate', 'upper', 'upper', 'zfill']` (con `'upper'` repetido). En Python 3.11.9 sale `['swapcase', 'title', 'translate', 'upper', 'zfill']`. La clave aclara que «varía según versión», pero el duplicado no es una variación real.
5. **Ej. 3: los flotantes no son exactos.** `1 - 0.91` es 0.08999999999999997 y `0.91 - 0.87 == 0.04` da `False`. La salida «0.09» y «0.04» es correcta solo porque `:.2f` redondea al imprimir. Para comparar, usar `math.isclose` o `round`. La comparación `accuracy_modelo_b > accuracy_modelo_a` sí es segura porque no pide igualdad.
6. **Ej. 10: `float(None)` no se atrapa.** Lanza `TypeError`, y el `except ValueError` de la solución lo deja pasar y el programa se detiene. Tampoco acepta coma decimal (`"0,87"` falla), algo habitual en datos capturados en español. Un `nan` sí queda rechazado, pero con el mensaje «fuera de rango». La versión de la sección 7.2 cubre estos casos.
7. **Ej. 7: `match` requiere Python 3.10 o superior.** Colab y Python 3.11.9 lo soportan; en una versión anterior da `SyntaxError`. La alternativa es `if` / `elif` encadenados, como indica la clave.
8. **Ej. 8: valores iniciales.** `mejor_accuracy = 0` funciona porque las accuracy están en [0, 1]; con una métrica que pueda ser negativa (pérdida, R²) habría que iniciar con `None` o con el primer elemento. Con una lista vacía, `len(...)` es 0 y el promedio da `ZeroDivisionError`.
9. **Ej. 11: archivo y codificación.** El modo `"w"` **sobrescribe** `historial.txt` en la carpeta de trabajo. La codificación por defecto depende del sistema operativo; si un nombre lleva tildes o `ñ`, conviene `encoding="utf-8"` al escribir y al leer. Lo leído es texto: hay que convertir con `float(...)` para operar.
10. **Ej. 12: dónde corre el `import`.** Con `python main.py` (incluso lanzado desde otra carpeta) funciona porque la carpeta del script entra en `sys.path`. En Colab o Jupyter esa regla no aplica de la misma forma: los dos archivos deben estar en la carpeta de trabajo del notebook (no se probó en Colab).
11. **Ej. 5: orden del conjunto.** Se imprime en un orden que puede cambiar entre ejecuciones; no sirve para comparar salidas literalmente. Comprobar con `len(...)` o con `sorted(modelos)`.

**Actividad en Colab**

12. **Sin validación de rango.** La tarea 2 dice «nota entre 0 y 5», pero no pide validar: un 7 imprimiría «Aprobó». Si se quiere, se añade `if not 0 <= nota <= 5` antes del `if nota >= 3.0` (➕). El umbral 3.0 sale del ejemplo de la guía, no de una regla explícita del curso.
