# Instrucciones para ejecutar Jupyter con Docker Compose

## 1. Verificar que Docker esté instalado

Abre una terminal y ejecuta:

```bash
docker --version
docker compose version
```

Si no aparecen versiones, instala Docker Desktop o Docker Engine antes de continuar.

## 2. Ubicarte en la carpeta del proyecto

```bash
cd C:\xampp\htdocs\web-uniminuto\NRC-70446-Fundamentos-para-IA
```

## 3. Construir la imagen del contenedor

```bash
docker compose build
```

Si quieres forzar una reconstrucción completa:

```bash
docker compose build --no-cache
```

## 4. Levantar el contenedor en segundo plano

```bash
docker compose up -d
```

## 5. Verificar que el contenedor esté corriendo

```bash
docker compose ps
```

## 6. Revisar los logs si hay errores

```bash
docker compose logs -f jupyter
```

## 7. Abrir Jupyter en el navegador

Abre esta URL:

```text
http://localhost:8888/lab
```

Como el contenedor usa token vacío, no debería pedir contraseña ni token.

## 8. Detener el contenedor

```bash
docker compose down
```

## 9. Reiniciar desde cero

```bash
docker compose down
docker compose up --build -d
```

## 10. Instalar paquetes desde requirements.txt

Si quieres instalar los paquetes definidos en `docker-requirements.txt` dentro del contenedor, ejecuta:

```bash
docker compose exec jupyter pip install -r /app/docker-requirements.txt
```

Si prefieres reconstruir el contenedor desde cero con esos paquetes:

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

## 11. Si quieres ejecutar comandos dentro del contenedor

```bash
docker compose exec jupyter bash
```

O instalar paquetes manualmente si es necesario:

```bash
docker compose exec jupyter pip install nombre-del-paquete
```

jupyterlab
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow-cpu
torch --index-url https://download.pytorch.org/whl/cpu

```bash
docker compose exec jupyter pip install -r /app/docker-requirements.txt
```
```bash
docker compose exec jupyter pip install tensorflow-cpu
```
```bash
docker compose exec jupyter pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 12. Importante

Si cambias el Dockerfile o requirements.txt, debes reconstruir la imagen antes de levantar el contenedor nuevamente:

```bash
docker compose down
docker compose build --no-cache
docker compose up -d
```

## 13. Archivos relevantes

- Dockerfile
- docker-compose.yml
- requirements.txt

Estos archivos controlan la imagen, el servicio y los paquetes de Python que se instalarán.
