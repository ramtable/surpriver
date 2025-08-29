# Stage 1: Copy Tailscale binaries from official image
FROM docker.io/tailscale/tailscale:stable AS tailscale

# Stage 2: Main Python app
FROM python:3.9-slim

# Install runtime deps
RUN apt-get update && apt-get install -y \
    iproute2 iputils-ping curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Tailscale binaries into PATH
COPY --from=tailscale /usr/local/bin/tailscaled /usr/local/bin/tailscaled
COPY --from=tailscale /usr/local/bin/tailscale /usr/local/bin/tailscale
RUN mkdir -p /var/run/tailscale /var/cache/tailscale /var/lib/tailscale

# Setup environment
WORKDIR /app
RUN cp /usr/local/bin/pip3.9 /usr/local/bin/pip3 && pip3 install --upgrade pip

# Install Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your app code (including entry_point.sh)
COPY . .

# Ensure entry_point.sh is executable
RUN chmod +x /app/entry_point.sh

VOLUME ["/app"]

ENTRYPOINT ["sh", "/app/entry_point.sh"]
CMD ["/app/entry_point.sh"]
