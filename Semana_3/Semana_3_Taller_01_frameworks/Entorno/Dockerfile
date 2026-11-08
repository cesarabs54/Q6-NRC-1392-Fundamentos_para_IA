
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# (opcional) dependencias de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Construir con el contexto en la carpeta del taller (la que contiene este directorio Entorno):
#   docker build -f Entorno/Dockerfile -t taller-01 .
#   docker run -p 8888:8888 taller-01
COPY Entorno/requirements.txt .

# No hay GPU real dentro del contenedor: evita que TensorFlow/PyTorch intenten
# inicializar CUDA (eso puede tumbar el kernel de Jupyter con un error de cuInit).
ENV CUDA_VISIBLE_DEVICES=-1

RUN pip install --no-cache-dir --upgrade pip \
 # torch CPU-only primero: evita descargar ~3 GB de paquetes nvidia-* que no se usan sin GPU.
 # Al estar ya instalado, el `torch` de requirements.txt no se vuelve a descargar.
 && pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Copia el material del estudiante de este taller (notebook y guía .md).
# Los archivos *_profesor.* no coinciden con este patrón y no se incluyen.
COPY Semana_3_taller_01.* ./

EXPOSE 8888

# Levanta Jupyter Notebook accesible desde el host en http://localhost:8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
