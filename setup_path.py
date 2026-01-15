from os.path import join, realpath, dirname
from pathlib import Path

BASE_PATH = dirname(realpath(__file__))
PARAMS_PATH = join(BASE_PATH, 'yaml')
RUN_PATH = join(BASE_PATH, 'run')
DATA_PATH = '/lotterlab/datasets/VCC'

# All results saved to this central location
OUTPUT_PATH = Path('/lotterlab/datasets/VCC/ML_OUTPUTS/SC_FM/eval_results')
EMBEDDINGS_PATH = OUTPUT_PATH / 'embeddings'