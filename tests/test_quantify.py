from pathlib import Path
import logging

from aicsimageio import AICSImage
import numpy as np
import pytest

from blimp.utils import equal_dims
from blimp.constants import blimp_config
import blimp.processing.quantify

from .helpers import _ensure_test_data  # noqa: F401, I252
from .helpers import _load_test_data

logger = logging.getLogger(__name__)


def test_quantify(_ensure_test_data):
    images = _load_test_data("synthetic_images")
