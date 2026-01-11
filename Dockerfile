# Usamos una base que ya tiene Python y CUDA (para la GPU)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Evitar que Python genere archivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /

# Instalar dependencias del sistema (por si acaso)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requisitos
COPY requirements.txt .

# Instalar las librerías de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar el resto de tu código (handler.py)
COPY . .

# El comando de arranque
CMD [ "python", "-u", "handler.py" ]
