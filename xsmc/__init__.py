import logging

from .segmentation import Segmentation
from .xsmc import XSMC

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(threadName)s %(message)s"
)
logging.getLogger(__name__).setLevel(logging.INFO)

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("xsmc")
except PackageNotFoundError:
    # package is not installed
    pass
