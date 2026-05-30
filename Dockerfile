# declario agent-runtime — Python + Playwright + Chromium.
#
# Base image already ships with Chromium + system deps + Python. That's
# the heaviest part of the install, so starting from a Playwright image
# keeps cold-build times bearable on Railway (~3 min instead of ~12).
FROM mcr.microsoft.com/playwright/python:v1.49.0-noble

WORKDIR /app

# The Microsoft playwright/python base image already ships with
# playwright + Chromium pre-installed in /usr/local. We install our
# extra Python deps with the SAME python so they land in the same
# site-packages — using /usr/bin/python (a different interpreter
# without playwright) would split the install and break browser-use.
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip install --no-cache-dir "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0" "pydantic>=2.8.0"

# Chromium is already baked into the base image, so `playwright install`
# is a no-op in the common case. Run it anyway to surface any version
# drift between our pip-installed playwright and the base's Chromium —
# fails loud rather than at first user request.
RUN python3 -m playwright install chromium

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
CMD ["sh", "-c", "python3 -m uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
