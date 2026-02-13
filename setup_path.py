import os
from os.path import join, realpath, dirname
from pathlib import Path

BASE_PATH = dirname(realpath(__file__))
PARAMS_PATH = os.environ.get('SCFM_EVAL_PARAMS_PATH', join(BASE_PATH, 'yaml'))
RUN_PATH = join(BASE_PATH, 'run')
DATA_PATH = '/lotterlab/datasets/VCC'

# All results saved to this central location (overridable for AB testing)
OUTPUT_PATH = Path(os.environ.get('SCFM_EVAL_OUTPUT_PATH',
                                  '/lotterlab/datasets/VCC/ML_OUTPUTS/SC_FM/eval_results'))
EMBEDDINGS_PATH = OUTPUT_PATH / 'embeddings'

# Custom temp directory on network storage (avoids filling local /tmp)
# This is used for large temporary files (h5ad, embeddings, etc.)
TEMP_PATH = OUTPUT_PATH / '.temp'
TEMP_PATH.mkdir(parents=True, exist_ok=True)