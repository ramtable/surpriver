# Stage 1: Build Tailscale
FROM alpine:3.19 as tailscale
RUN apk add --no-cache curl ca-certificates && \
    curl -fsSL https://pkgs.tailscale.com/stable/tailscale_1.66.4_amd64.tgz | tar xz -C /tmp && \
    mkdir -p /out && \
    cp /tmp/tailscale* /out/ && \
    cp /tmp/tailscaled /out/

# Stage 2: Main Python app
FROM python:3.9

# Copy Tailscale binaries from the previous stage
COPY --from=tailscale /out/tailscale /usr/local/bin/tailscale
COPY --from=tailscale /out/tailscaled /usr/local/bin/tailscaled

# Setup environment
RUN  cp /usr/local/bin/pip3.9 /usr/local/bin/pip3  # reenable pip3
RUN pip3 install --upgrade pip
WORKDIR /usr/src/app

# Install requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/usr/src/app"]

# Start Tailscale and your app (adjust as needed)
ENTRYPOINT [ "sh", "/usr/src/app/entry_point.sh" ]
CMD ["/usr/src/app/entry_point.sh"]
