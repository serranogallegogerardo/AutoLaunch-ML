"""Respaldo y restauración de MLflow (SQLite) y de los artefactos del modelo.

Uso:
  python backup.py respaldar            # crea backups/AAAAMMDD-HHMMSS con manifiesto SHA-256
  python backup.py restaurar <carpeta>  # verifica hashes y restaura
Retención: se conservan las últimas RETENCION copias.
"""
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGENES = [HERE / "mlruns.db", HERE / "model" / "artifacts"]
DESTINO = HERE / "backups"
RETENCION = 14


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def archivos(base: Path):
    return [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file())


def respaldar() -> Path:
    dst = DESTINO / time.strftime("%Y%m%d-%H%M%S")
    dst.mkdir(parents=True)
    manifest = {}
    for src in ORIGENES:
        for f in archivos(src):
            rel = f.relative_to(HERE)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / rel)
            manifest[str(rel).replace("\\", "/")] = sha256(f)
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    copias = sorted(p for p in DESTINO.iterdir() if p.is_dir())
    for vieja in copias[:-RETENCION]:
        shutil.rmtree(vieja)
    return dst


def restaurar(carpeta: Path) -> float:
    t0 = time.perf_counter()
    manifest = json.loads((carpeta / "manifest.json").read_text(encoding="utf-8"))
    for rel, h in manifest.items():
        if sha256(carpeta / rel) != h:
            raise SystemExit(f"Hash inválido en la copia: {rel}")
    for rel in manifest:
        (HERE / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(carpeta / rel, HERE / rel)
    for rel, h in manifest.items():
        assert sha256(HERE / rel) == h
    return time.perf_counter() - t0


if __name__ == "__main__":
    if sys.argv[1:2] == ["respaldar"]:
        print("Copia creada:", respaldar())
    elif sys.argv[1:2] == ["restaurar"]:
        print(f"Restauración verificada en {restaurar(Path(sys.argv[2])):.2f} s")
    else:
        print(__doc__)
