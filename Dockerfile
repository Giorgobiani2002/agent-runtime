# declario agent-runtime — Python + Playwright + Chromium.
#
# Base image already ships with Chromium + system deps + Python. That's
# the heaviest part of the install, so starting from a Playwright image
# keeps cold-build times bearable on Railway (~3 min instead of ~12).
FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

WORKDIR /app

# Install Python deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0" "pydantic>=2.8.0"

# Browser-use needs Chromium specifically; the Playwright base ships it
# but make sure browser-use's expected install path is populated.
RUN python -m playwright install chromium --with-deps

# Now copy the app code.
COPY . .

# Default to headless inside the container (no virtual display).
ENV AGENT_HEADLESS=true \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

EXPOSE 8080

# Railway runs whatever's in package.json or what we set as start cmd.
# Bind 0.0.0.0 so Railway's healthcheck can reach us.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
