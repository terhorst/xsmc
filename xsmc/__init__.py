import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(threadName)s %(message)s')
logging.getLogger(__name__).setLevel(logging.INFO)

from .xsmc import XSMC
from .segmentation import Segmentation
