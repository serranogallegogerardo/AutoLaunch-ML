from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_dashboard_renderiza_sin_errores():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("PulseML" in t.value for t in at.title)
    assert len(at.tabs) == 2
