
from .aug import AugModel
from .mmn import MMNModel
from .sample import SampleModel

META_MODEL_DICT = {
    'aug': AugModel,
    'mmn': MMNModel,
    'sample': SampleModel
}