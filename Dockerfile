<<<<<<< HEAD
FROM python:3.10

WORKDIR /app

COPY . .

# Install dependencies
RUN pip install --no-cache-dir numpy

# If you want GUI in Docker, you need extra X11 setup (optional)
# For terminal-only RL simulation, above is enough

=======
FROM python:3.10

WORKDIR /app

COPY . .

# Install dependencies
RUN pip install --no-cache-dir numpy

# If you want GUI in Docker, you need extra X11 setup (optional)
# For terminal-only RL simulation, above is enough

>>>>>>> 85e45758f974eedc1504cf54d6431bd7544ce47b
CMD ["python", "run.py"]