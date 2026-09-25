"""Descarga el dataset público de la UCI y verifica su hash SHA-256."""
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/498/incident+management+process+enriched+event+log.zip"
DEST = Path(__file__).resolve().parents[1] / "data" / "incident_event_log.csv"
SHA256 = sys.argv[1] if len(sys.argv) > 1 else None

if not DEST.exists():
    DEST.parent.mkdir(exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    name = next(n for n in z.namelist() if n.endswith(".csv"))
    DEST.write_bytes(z.read(name))

digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
print(DEST, digest)
if SHA256 and digest != SHA256:
    sys.exit("El hash no coincide con el esperado")
