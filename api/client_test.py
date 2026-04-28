"""
client_test.py — Skrip untuk mencoba semua endpoint FastAPI
Python  : 3.10.9
Jalankan setelah server aktif: python client_test.py
"""

import sys
import json
import requests

BASE_URL = "http://127.0.0.1:8000"


def print_section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


def print_response(resp):
    print(f"Status Code : {resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(resp.text)


# ── 1. Root ──────────────────────────────────────────────────
print_section("1. GET / — Info API")
resp = requests.get(f"{BASE_URL}/")
print_response(resp)

# ── 2. Health Check ──────────────────────────────────────────
print_section("2. GET /health — Health Check")
resp = requests.get(f"{BASE_URL}/health")
print_response(resp)

# ── 3. Model Info ────────────────────────────────────────────
print_section("3. GET /model/info — Info Model")
resp = requests.get(f"{BASE_URL}/model/info")
print_response(resp)

# ── 4. Prediksi Satu Gambar ───────────────────────────────────
print_section("4. POST /predict — Prediksi Satu Gambar")

img_path = sys.argv[1] if len(sys.argv) > 1 else None

if img_path:
    with open(img_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/predict",
            files={"file": (img_path, f, "image/jpeg")}
        )
    print_response(resp)
else:
    # Buat gambar dummy 128x128 (merah) untuk test
    import io
    from PIL import Image
    import numpy as np

    dummy = Image.fromarray(
        (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    )
    buf = io.BytesIO()
    dummy.save(buf, format="JPEG")
    buf.seek(0)

    resp = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("test_image.jpg", buf, "image/jpeg")}
    )
    print_response(resp)
    print("\n[INFO] Gambar dummy digunakan. Kirim path gambar nyata:")
    print("       python client_test.py path/ke/gambar.jpg")

# ── 5. Prediksi Batch ─────────────────────────────────────────
print_section("5. POST /predict/batch — Prediksi Batch (3 gambar dummy)")

import io
from PIL import Image
import numpy as np

files = []
for i in range(3):
    dummy = Image.fromarray(
        (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    )
    buf = io.BytesIO()
    dummy.save(buf, format="JPEG")
    buf.seek(0)
    files.append(("files", (f"dummy_{i}.jpg", buf, "image/jpeg")))

resp = requests.post(f"{BASE_URL}/predict/batch", files=files)
print_response(resp)

print("\n✅ Semua test selesai!")
