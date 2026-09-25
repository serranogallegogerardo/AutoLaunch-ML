"""Estilo común de figuras."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline import ROOT  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE = "#002060"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 11, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)
