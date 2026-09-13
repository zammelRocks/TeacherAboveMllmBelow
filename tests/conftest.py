import numpy as np
import pytest
from PIL import Image, ImageDraw

from kinematics_grading.config import load_settings


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture
def rubric_ex2():
    return {
        "max_score": 10,
        "rules": [
            {"id": "forward", "points": 3, "quantity": "dfdx", "relation": "gt", "expected": 0.1},
            {"id": "parking", "points": 4, "quantity": "dfdx", "relation": "eq", "expected": 0},
            {"id": "reverse", "points": 3, "quantity": "dfdx", "relation": "lt", "expected": -0.1},
        ],
    }


def make_synthetic_plot(size: int = 512, kind: str = "line") -> Image.Image:
    """A minimal synthetic 'graph' image: axes + a blue line, good enough to
    exercise curve_mask/detect_plot_box without needing real corpus images."""
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    left, top, right, bottom = 72, 55, 470, 430
    d.line((left, bottom, right, bottom), fill=(20, 20, 20), width=2)
    d.line((left, bottom, left, top), fill=(20, 20, 20), width=2)

    if kind == "line":
        d.line([(left, bottom - 20), (right, top + 20)], fill=(31, 119, 180), width=4)
    elif kind == "flat":
        mid = (top + bottom) // 2
        d.line([(left, mid), (right, mid)], fill=(31, 119, 180), width=4)
    return img


@pytest.fixture
def synthetic_increasing_image():
    return make_synthetic_plot(kind="line")


@pytest.fixture
def synthetic_flat_image():
    return make_synthetic_plot(kind="flat")
