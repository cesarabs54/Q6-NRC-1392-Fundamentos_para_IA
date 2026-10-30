
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# (opcional) dependencias de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# No hay GPU real dentro del contenedor: evita que TensorFlow/PyTorch intenten
# inicializar CUDA (eso puede tumbar el kernel de Jupyter con un error de cuInit).
ENV CUDA_VISIBLE_DEVICES=-1

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r docker-requirements.txt \
 # torch CPU-only: evita descargar ~3 GB de paquetes nvidia-* que no se usan sin GPU
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copia el material del taller (archivos .md de esta misma carpeta)
COPY *.md .

EXPOSE 8888

# Levanta Jupyter Notebook accesible desde el host en http://localhost:8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]