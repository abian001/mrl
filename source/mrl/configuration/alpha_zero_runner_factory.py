import yaml

from mrl.alpha_zero.alpha_zero import AlphaZero
from mrl.alpha_zero.context import (
    EvaluationContext,
    EvaluationPolicies,
    HDF5AlphaZeroContext,
    InMemoryAlphaZeroContext,
    UnionAlphaZeroContext,
)
from mrl.alpha_zero.distributed_alpha_zero import DistributedAlphaZero
from mrl.alpha_zero.mcts import MCTSGame
from mrl.configuration.alpha_zero_configuration import (
    AlphaZeroConfiguration,
    HDF5AlphaZeroConfiguration,
    ModelTestConfiguration,
)
from mrl.configuration.alpha_zero_runner_configuration import AlphaZeroRunnerConfiguration
from mrl.configuration.player_utils import validate_player
from mrl.configuration.runner_factories import (
    make_gui,
    make_mcts_game,
    make_policy,
    make_stdin_policy,
    make_trainable_oracle,
)


class AlphaZeroRunnerFactory:

    def make_context(
        self,
        configuration: AlphaZeroConfiguration,
    ) -> UnionAlphaZeroContext:
        game = make_mcts_game(configuration.game_configuration)
        self._validate_report_generator_players(configuration.report_generator, game)
        self._validate_output_size(configuration, game)
        oracle = make_trainable_oracle(configuration.oracle_configuration, game)
        evaluation_oracle = make_trainable_oracle(configuration.oracle_configuration, game)
        evaluation_policies = EvaluationPolicies(
            lead = make_policy(
                configuration.evaluation.policy_configuration,
                game = game,
                oracle = oracle,
            ),
            opponent = make_policy(
                configuration.evaluation.policy_configuration,
                game = game,
                oracle = evaluation_oracle,
            ),
        )
        evaluation_context = EvaluationContext(
            episodes = configuration.evaluation.episodes,
            max_old_models = configuration.evaluation.max_old_models,
            oracle = evaluation_oracle,
            policies = evaluation_policies,
        )
        base_arguments = {
            'game': game,
            'oracle_configuration': configuration.oracle_configuration,
            'oracle': oracle,
            'trainer': configuration.trainer,
            'collector': configuration.collector,
            'number_of_epochs': configuration.number_of_epochs,
            'report_generator': configuration.report_generator,
            'config_file_path': configuration.config_file_path,
            'workspace_path': configuration.workspace_path,
            'evaluation': evaluation_context,
            'oracle_file_path': configuration.oracle_file_path,
        }
        if isinstance(configuration.data, HDF5AlphaZeroConfiguration):
            return self._make_hdf5_context(
                configuration,
                base_arguments,
            )
        return InMemoryAlphaZeroContext(**base_arguments)

    def _make_hdf5_context(
        self,
        configuration: AlphaZeroConfiguration,
        base_arguments: dict,
    ) -> HDF5AlphaZeroContext:
        assert isinstance(configuration.data, HDF5AlphaZeroConfiguration)
        context = HDF5AlphaZeroContext(
            **base_arguments,
            hdf5_path_prefix = configuration.data.hdf5_path_prefix,
            server_hostname = configuration.data.server_hostname,
            server_port = configuration.data.server_port,
        )
        if not context.config_file_path.exists():
            with open(context.config_file_path, "w", encoding = "UTF-8") as yaml_file:
                yaml.dump(configuration.model_dump(by_alias = True), yaml_file)
        return context

    def make_alpha_zero(
        self,
        context: UnionAlphaZeroContext,
    ) -> AlphaZero | DistributedAlphaZero:
        if context.type == "InMemory":
            return AlphaZero(context)
        if context.type == "HDF5":
            return DistributedAlphaZero(context)
        raise TypeError(f"Unsupported alpha zero configuration type {context.type}.")

    def make_manual_player(
        self,
        configuration: AlphaZeroRunnerConfiguration,
        context: UnionAlphaZeroContext,
    ):
        return validate_player(
            context.game,
            configuration.manual_play.manual_player,
        )

    def make_autonomous_policy(
        self,
        configuration: AlphaZeroRunnerConfiguration,
        context: UnionAlphaZeroContext,
    ):
        return make_policy(
            configuration.manual_play.policy_configuration,
            game = context.game,
            oracle = context.oracle,
        )

    def make_stdin_policy_for_runner(
        self,
        configuration: AlphaZeroRunnerConfiguration,
        _context: UnionAlphaZeroContext,
        manual_player,
    ):
        return make_stdin_policy(
            configuration.alpha_zero.game_configuration,
            configuration.stdin_configuration,
            player = manual_player,
        )

    def make_manual_gui(
        self,
        configuration: AlphaZeroRunnerConfiguration,
        _context: UnionAlphaZeroContext,
    ):
        return make_gui(
            configuration.alpha_zero.game_configuration,
            configuration.gui_configuration,
        )

    def _validate_report_generator_players(
        self,
        report_generator: ModelTestConfiguration,
        game: MCTSGame,
    ) -> None:
        report_generator.oracle_led_players = tuple(
            validate_player(game, player)
            for player in report_generator.oracle_led_players
        )

    def _validate_output_size(
        self,
        configuration: AlphaZeroConfiguration,
        game: MCTSGame,
    ) -> None:
        any_perspective = next(iter(game.get_perspectives().values()))
        if not all(
            any_perspective.action_space_dimension == perspective.action_space_dimension
            for perspective in game.get_perspectives().values()
        ):
            raise TypeError(
                f'Perspectives of game {type(game)} return different action '
                'space sizes. This implementation of alpha zero only '
                'supports games whose perspectives have the same action space.'
            )
        output_size = configuration.oracle_configuration.output_size
        if output_size is not None and any_perspective.action_space_dimension != output_size:
            raise TypeError(
                'Mismatched output sizes between the game and the model. The game '
                f'output space size is {any_perspective.action_space_dimension}. '
                f'The model output size is {output_size}.'
            )
