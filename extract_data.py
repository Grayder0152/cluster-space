import os
from typing import Optional

import pandas as pd

from settings import SAMPLE_DATASETS_DIR


def extract(file_name: str, columns: Optional[list[str]] = None, delimiter: str = ','):
    df = pd.read_csv(os.path.join(SAMPLE_DATASETS_DIR, file_name), delimiter=delimiter)
    if columns is not None:
        df = df[columns]
    return df
