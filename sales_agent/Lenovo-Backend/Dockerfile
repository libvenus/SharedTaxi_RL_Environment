FROM python:3.12-slim

# psql client is needed by entrypoint.sh to run migrations
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    --no-cache-dir \
    -r requirements.txt

COPY . .

RUN chmod +x /app/entrypoint.sh

EXPOSE 9100

ENTRYPOINT ["/app/entrypoint.sh"]
