FROM python:3.12-slim

# Install uv (trusted-host avoids the corporate TLS interception issue)
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    uv

WORKDIR /app

# 1) Install ONLY dependencies first (better layer caching, no project build).
#    --no-install-project skips building the local app package, so README.md
#    and the package source are not needed at this stage.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project \
    --allow-insecure-host pypi.org \
    --allow-insecure-host files.pythonhosted.org

# 2) Now copy the application code.
COPY . .

# Use the venv uv created (no re-sync at runtime).
ENV PATH="/app/.venv/bin:$PATH"

# orchestrator_api.py binds 0.0.0.0:9091
EXPOSE 9091

CMD ["python", "orchestrator_api.py"]