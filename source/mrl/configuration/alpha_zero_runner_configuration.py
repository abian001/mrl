from dataclasses import dataclass
from pathlib import Path
from typing import Generic
from pydantic import BaseModel, Field
from mrl.game.game import Player, Policy
from mrl.alpha_zero.mcts import MCTSGame
from mrl.alpha_zero.oracle import Oracle
from mrl.configuration.factory import ObjectConfiguration
from mrl.configuration.game_factories import make_policy, make_stdin_policy
from mrl.configuration.gui_factory import make_gui
from mrl.configuration.player_utils import validate_player
from mrl.configuration.alpha_zero_configuration import (
    AlphaZeroConfiguration,
    UnionAlphaZeroConfiguration
)
from mrl.tkinter_gui.gui import Gui


@dataclass
class ManualPlaySpecification(Generic[Player]):
    manual_player: Player
    autonomous_policy: Policy
    gui_configuration: ObjectConfiguration | None
    stdin_configuration: ObjectConfiguration | None


@dataclass
class AlphaZeroRunnerSpecification(Generic[Player]):
    alpha_zero: UnionAlphaZeroConfiguration
    manual_play: ManualPlaySpecification[Player]
    _gui: Gui | None = None
    _stdin_policy: Policy | None = None

    @property
    def gui(self) -> Gui:
        if self._gui is None:
            self._gui = make_gui(
                self.alpha_zero.game_configuration,
                self.manual_play.gui_configuration
            )
        return self._gui

    @property
    def game(self) -> MCTSGame:
        return self.alpha_zero.game

    @property
    def oracle(self) -> Oracle:
        return self.alpha_zero.oracle

    @property
    def oracle_file_path(self) -> Path:
        return self.alpha_zero.oracle_file_path

    @property
    def autonomous_policy(self) -> Policy:
        return self.manual_play.autonomous_policy

    @property
    def manual_player(self) -> Player:
        return self.manual_play.manual_player

    @property
    def stdin_policy(self) -> Policy:
        if self._stdin_policy is None:
            self._stdin_policy = make_stdin_policy(
                self.alpha_zero.game_configuration,
                self.manual_play.stdin_configuration,
                self.manual_play.manual_player
            )
        return self._stdin_policy


class ManualPlayConfiguration(BaseModel):
    manual_player: str | None = None  # None: will be a randomly choosen player
    policy_configuration: ObjectConfiguration = Field(
        alias = 'autonomous_policy',
        default_factory = lambda: ObjectConfiguration(name = 'DeterministicOraclePolicy')
    )
    gui_configuration: ObjectConfiguration | None = Field(alias = 'gui', default = None)
    stdin_configuration: ObjectConfiguration | None = Field(alias = 'stdin_policy', default = None)


def make_alpha_zero_runner_specification(data: dict) -> AlphaZeroRunnerSpecification:
    alpha_zero = AlphaZeroConfiguration.model_validate({'alpha_zero': data}).alpha_zero
    manual_play_config = ManualPlayConfiguration.model_validate(data.get('manual_play', {}))
    policy = make_policy(
        manual_play_config.policy_configuration,
        extra_arguments = {
            'game': alpha_zero.game,
            'oracle': alpha_zero.oracle
        }
    )
    return AlphaZeroRunnerSpecification(
        alpha_zero = alpha_zero,
        manual_play = ManualPlaySpecification(
            manual_player = validate_player(alpha_zero.game, manual_play_config.manual_player),
            autonomous_policy = policy,
            gui_configuration = manual_play_config.gui_configuration,
            stdin_configuration = manual_play_config.stdin_configuration
        )
    )
