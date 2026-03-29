from pydantic import BaseModel, Field, model_validator

from mrl.configuration.alpha_zero_configuration import (
    AlphaZeroConfiguration,
    ManualPlayConfiguration,
)
from mrl.configuration.factory import ObjectConfiguration


class AlphaZeroRunnerConfiguration(BaseModel):
    alpha_zero: AlphaZeroConfiguration
    manual_play: ManualPlayConfiguration = Field(default_factory = ManualPlayConfiguration)
    gui_configuration: ObjectConfiguration | None = Field(alias = 'gui', default = None)
    stdin_configuration: ObjectConfiguration | None = Field(alias = 'stdin_policy', default = None)

    @model_validator(mode = 'before')
    @classmethod
    def _wrap_data(cls, data):
        if not isinstance(data, dict) or 'alpha_zero' in data:
            return data
        return {
            'alpha_zero': {
                key: value
                for (key, value) in data.items()
                if key not in {'manual_play', 'gui', 'stdin_policy'}
            },
            'manual_play': data.get('manual_play', {}),
            'gui': data.get('gui'),
            'stdin_policy': data.get('stdin_policy'),
        }
