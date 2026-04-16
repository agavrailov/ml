#!/usr/bin/env bash
# Deploy systemd services from this directory.
# Run as root (or with sudo) on the Oracle VM.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SYSTEMD_DIR="$REPO_DIR/deploy/systemd"

echo "Copying service files to /etc/systemd/system/ ..."
for svc in xvfb ib-gateway ml-lstm-trader ml-lstm-ui cloudflared; do
    cp "$SYSTEMD_DIR/${svc}.service" /etc/systemd/system/
    echo "  installed ${svc}.service"
done

systemctl daemon-reload

echo "Enabling services ..."
systemctl enable xvfb ib-gateway ml-lstm-trader ml-lstm-ui cloudflared

echo "Starting xvfb and ib-gateway (trader + UI start after Gateway is ready) ..."
systemctl start xvfb
systemctl start ib-gateway

echo ""
echo "Wait ~90s for IB Gateway to finish logging in, then run:"
echo "  sudo systemctl start ml-lstm-trader ml-lstm-ui cloudflared"
echo "  sudo systemctl status ml-lstm-trader"
