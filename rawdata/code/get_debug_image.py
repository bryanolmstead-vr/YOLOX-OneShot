#!/usr/bin/env python3
"""
Fetch a debug image over XMLRPC and save it as a PNG.
created using gpt-5.2-codex
Ben Hogervorst, 02.10.2026
"""

import argparse
import xmlrpc.client

import numpy as np

try:
    import cv2
except Exception as exc:
    raise SystemExit("cv2 is required (pip install opencv-python)") from exc


def fetch_debug_image(host: str, port: int):
    url = f"http://{host}:{port}/RPC2"
    rpc = xmlrpc.client.ServerProxy(url, allow_none=True)
    shape, raw = rpc.get_debug_image()
    if shape == (0, 0) or raw == 0:
        return None
    data = raw.data if hasattr(raw, "data") else raw
    img = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    return img


def main():
    parser = argparse.ArgumentParser(description="Fetch debug image over XMLRPC.")
    parser.add_argument("--host", default="127.0.0.1", help="VIM host address")
    parser.add_argument("--port", type=int, default=8000, help="XMLRPC port")
    parser.add_argument("--out", default="debug_image.png", help="Output PNG filename")
    args = parser.parse_args()

    img = fetch_debug_image(args.host, args.port)
    if img is None:
        raise SystemExit("No image available (shape was empty).")

    ok = cv2.imwrite(args.out, img)
    if not ok:
        raise SystemExit(f"Failed to write image to {args.out}")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
