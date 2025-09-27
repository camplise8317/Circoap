# Usa una imagen base de Python ligera y oficial.
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor.
WORKDIR /app

# Copia primero el archivo de requerimientos.
# Esto aprovecha el caché de capas de Docker: si los requerimientos no cambian,
# no se volverán a instalar en cada build, haciendo el proceso más rápido.
COPY requirements.txt ./requirements.txt

# Instala las dependencias de Python.
# --no-cache-dir reduce el tamaño de la imagen.
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación al directorio de trabajo.
COPY . .

# Expone el puerto por defecto que usa Streamlit.
EXPOSE 8501

# El comando para ejecutar la aplicación cuando se inicie el contenedor.
# --server.address=0.0.0.0 es crucial para que sea accesible desde fuera del contenedor.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
