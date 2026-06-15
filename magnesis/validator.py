import os
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class GNDataValidator(ABC):
    @abstractmethod
    def validate(self, geomagnetic_df) -> None:
        raise NotImplementedError()


class GNDataValidatorImpl(GNDataValidator):
    # def validate
    def validate(self, geomagnetic_df: pd.DataFrame) -> None:
        pass

    def is_omni_file(self, file: str):
        if not os.path.exists(path=file):
            raise FileNotFoundError(f"File not founded: {file}")
        # TODO: доделать проверки
