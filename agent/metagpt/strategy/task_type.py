from enum import Enum
from pydantic import BaseModel

from metagpt.prompts.task_type import (
    DATA_PREPROCESS_PROMPT,
    EDA_PROMPT,
    ECONOMETRIC_ALGORITHM_PROMPT,
)

class TaskTypeDef(BaseModel):
    name: str
    desc: str = ""
    guidance: str = ""

class TaskType(Enum):
    """By identifying specific types of tasks, we can inject human priors (guidance) to help task solving"""

    EDA = TaskTypeDef(
        name="eda",
        desc="For performing exploratory data analysis",
        guidance=EDA_PROMPT,
    )
    DATA_PREPROCESS = TaskTypeDef(
        name="data preprocessing",
        desc="For preprocessing dataset when datasets are found unclean or there are clear instructions from user",
        guidance=DATA_PREPROCESS_PROMPT,
    )

    ECONOMETRIC_ALGORITHM = TaskTypeDef(
        name="econometric algorithm",
        desc="For matching and applying an econometric algorithm tool.",
        guidance=ECONOMETRIC_ALGORITHM_PROMPT,
    )

    OTHER = TaskTypeDef(name="other", desc="Any tasks not in the defined categories")

    @property
    def type_name(self):
        return self.value.name

    @classmethod
    def get_type(cls, type_name):
        for member in cls:
            if member.type_name == type_name:
                return member.value
        return None