FROM python:3.10-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)


RUN pip install --no-cache-dir \
    torch==2.1.2+cpu torchvision==0.16.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html
    
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose HF default port
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
