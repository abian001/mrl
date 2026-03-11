from dataclasses import dataclass
from typing import Generic
import random
from pydantic_core import core_schema
from mrl.game.game import Player, Game, State


@dataclass
class ChoiceFunction(Generic[Player]):
    choices: str | tuple[str] = 'random'
    _validated_choices: tuple[Player] | None = None

    def __call__(self, game: Game) -> Player:
        if self._validated_choices is None:
            self._validated_choices = self._do_validate_choices(game)
        if len(self._validated_choices) == 1:
            return self._validated_choices[0]
        return random.choice(self._validated_choices)  # type: ignore[arg-type]

    def _do_validate_choices(self, game: Game) -> tuple[Player]:
        if isinstance(self.choices, str):
            if self.choices == 'random':
                return game.get_players()
            return (validate_player(game, self.choices),)
        return tuple((validate_player(game, p) for p in self.choices))

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        """Allows creating a choice function from a list of strings"""
        _, _ = source, handler
        return core_schema.no_info_before_validator_function(
            ChoiceFunction,
            core_schema.any_schema(),
            serialization = core_schema.plain_serializer_function_ser_schema(
                lambda choice_function: choice_function.choices
            )
        )


def validate_player(game: Game[State, Player], player: str | None) -> Player:
    if player is None:
        return random.choice(game.get_players())

    validated_player = next((
        _player for _player in game.get_players()
        if str(_player) == player
    ), None)
    if validated_player is None:
        raise TypeError(
            f"Invalid player {player}. Allowed players are: "
            f"{game.get_players()}."
        )
    return validated_player
