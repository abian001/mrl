###################
 Alpha Zero runner
###################

The AlphaZero runner is a command-line utility for training and testing
AlphaZero models. It can be used to:

-  train a new AlphaZero model or resume training an existing model;
-  evaluate a trained model against a random policy;
-  play against the trained model using either the terminal or the GUI.

The runner can be invoked from the command line as follows:

.. code:: bash

   run_alpha_zero examples/tic_tac_toe_alpha_zero.yaml --mode train

There are six available modes:

-  ``train``: train a new AlphaZero model. This mode prevents
   overwriting existing models.
-  ``overwrite``: train a new AlphaZero model and overwrite any existing
   model files.
-  ``resume``: resume training from an existing model.
-  ``evaluate``: evaluate the trained model against a random policy.
-  ``terminal``: play against the trained model using the terminal
   interface.
-  ``gui``: play against the trained model using the graphical
   interface.

***********************
 Example configuration
***********************

The following YAML configuration illustrates a complete setup. The
individual sections are described below.

.. code:: yaml

   game:
       name: MCTSTicTacToe
       first_player: random
   type: HDF5
   oracle:
       name: OpenSpielMLP
       capacity:
           input_size: 18
           output_size: 9
           nn_depth: 2
           nn_width: 2
       file_path: tic_tac_toe_alpha_zero
   collector:
       number_of_processes: 1
       mcts:
           number_of_simulations: 1
           pucb_constant: 1.0
           discount_factor: 1.0
           dirichlet_alpha: 0.3
           dirichlet_weight: 0.25
       max_buffer_length: 100
       number_of_episodes: 1
       temperature_schedule:
           - [0, 1.0]
   trainer:
       batch_size: 32
       max_training_epochs: 1
       early_stop_loss: 1e-3
       learning_rate: 1e-3
       loading_workers: 1
   report_generator:
       number_of_tests: 100
       buckets:
           - [-inf, 0.25]
           - [0.25, 0.75]
           - [0.75, +inf]
       policies:
           X:
               name: RandomPolicy
           O:
               name: DeterministicOraclePolicy
               oracle: TrainedOracle
   number_of_epochs: 1
   evaluation:
       episodes: 100
       max_old_models: 10
       uncertainty_penalty_coefficient: 3.0
       discount_factor: 1.0
       policy:
           name: DeterministicOraclePolicy
       true_skill:
           mu: 25.0
           sigma: 8.333
           beta: 1.476
           tau: 0.0
           draw_probability: 0.1
   hdf5_path_prefix: tic_tac_toe_data
   server_hostname: 127.0.0.1
   server_port: 8888
   workspace_path: test_workspace
   manual_play:
       manual_player: 'O'
       autonomous_policy:
           name: DeterministicOraclePolicy
   stdin_policy:
       name: TicTacToeStdin
       module: mrl.tic_tac_toe.game
   gui:
       name: make_gui
       module: mrl.tic_tac_toe.tkinter_gui

******
 Game
******

.. code:: yaml

   game:
       name: MCTSTicTacToe
       first_player: random

This section describes the game configuration. It follows the same
structure used by the ``game_runner``. See the :ref:`game
<game_runner_game>` for details.

*****************
 Alpha Zero Type
*****************

.. code:: yaml

   type: HDF5

   type: InMemory

This setting determines how training data is stored.

-  ``InMemory``: all training data is kept in memory, even when multiple
   training processes are used.
-  ``HDF5``: training data is stored in shared HDF5 files.

The HDF5 mode is useful when the dataset is too large to fit in memory.

********
 Oracle
********

.. code:: yaml

   oracle:
       name: OpenSpielMLP
       capacity:
           input_size: 18
           output_size: 9
           nn_depth: 2
           nn_width: 2
       file_path: tic_tac_toe_alpha_zero

This section defines the model being trained.

It includes:

-  the name of the trainable oracle class;
-  optional parameters required by the oracle constructor;
-  a ``file_path`` specifying where the model parameters are saved.

For ``run_alpha_zero``, the configured object must not be just a generic
``Oracle``. It must be a **TrainableOracle**, meaning an oracle whose
parameters can also be saved and restored during training and model
selection.

**********************
 Experience collector
**********************

.. code:: yaml

   collector:
       number_of_processes: 1
       mcts:
           number_of_simulations: 1
           pucb_constant: 1.0
           discount_factor: 1.0
           dirichlet_alpha: 0.3
           dirichlet_weight: 0.25
       max_buffer_length: 100
       number_of_episodes: 1
       temperature_schedule:
           - [0, 1.0]
           - [100, 0.5]
           - [200, 0.1]

This section configures the processes responsible for generating
training experience.

-  ``number_of_processes``: number of parallel simulation processes.

-  ``mcts``: parameters used by the MCTS policy during training. See the
   :ref:`MCTSPolicy <mcts_policy>` for details.

-  ``max_buffer_length``: maximum number of experience samples stored by
   each process. When this threshold is reached, older data is discarded
   (``InMemory`` mode) or written to disk (``HDF5`` mode).

-  ``number_of_episodes``: number of games simulated per training epoch.

Experience is generated using the ``NonDeterministicMCTSPolicy``.

The ``temperature_schedule`` controls how the exploration temperature
changes over time. It is defined as a list of pairs:

``[epoch_number, temperature_value]``

When the training epoch reaches ``epoch_number``, the temperature is
updated to the specified value.

See the :ref:`Non Deterministic MCTSPolicy <nondet_mcts_policy>` for
details about the ``temperature`` parameter.

*********
 Trainer
*********

.. code:: yaml

   trainer:
       batch_size: 32
       max_training_epochs: 1
       early_stop_loss: 1e-3
       learning_rate: 1e-3
       loading_workers: 1

This section defines the neural network training parameters.

-  ``batch_size``: number of experience samples processed in each
   training batch.
-  ``max_training_epochs``: maximum number of passes over the training
   data per training epoch.
-  ``early_stop_loss``: threshold used to stop training early if the
   loss falls below this value.
-  ``learning_rate``: learning rate used by the optimizer.
-  ``loading_workers``: number of worker threads used to prepare
   training data.

******************
 Report generator
******************

.. code:: yaml

   report_generator:
       number_of_tests: 100
       buckets:
           - [-inf, 0.25]
           - [0.25, 0.75]
           - [0.75, +inf]
       policies:
           X:
               name: RandomPolicy
           O:
               name: DeterministicOraclePolicy
               oracle: TrainedOracle

This section defines how evaluation results are reported.

During training, whenever a model is considered an improvement over its
predecessor (see `evaluation <alpha_zero_evaluation>`_), it is also
evaluated against a random policy. Its performance against the random
policy is then reported in the standard output.

-  ``number_of_tests``: number of evaluation games to run.

-  ``buckets``: payoff ranges used to group evaluation results.

-  ``oracles``: optional named oracles available only to the report
   generator.

-  ``shared_policies``: optional named reusable policy configurations.

-  ``policies``: player-to-policy mapping, in the same style as the
   ``game_runner`` configuration. See the ``Policies``, ``Shared
   policies``, and ``Shared oracles`` sections in :doc:`game_runner`.

The only difference from the ``game_runner`` format is that the special
oracle name ``TrainedOracle`` is also available. It refers to the main
AlphaZero oracle currently being trained.

Players whose effective policy uses ``oracle: TrainedOracle`` are the
players whose results are reported.

If a player is omitted from ``policies``, the runner uses a default
policy:

.. code:: yaml

   name: DeterministicOraclePolicy
   oracle: TrainedOracle

The report is generated every time a better model has been trained. When
a report is generated, the configured policies play ``number_of_tests``
games. For each game, the payoff of each observed player determines
which bucket that result belongs to. The final report then summarizes,
for each observed player, how many games fell into each bucket.

*************
 Manual play
*************

.. code:: yaml

   manual_play:
       manual_player: 'O'
       autonomous_policy:
           name: DeterministicOraclePolicy

This section configures manual play against the trained model.

-  ``manual_player`` specifies which player is controlled by the user.
-  ``autonomous_policy`` defines the policy used by the other players.

If the selected policy requires an oracle, it will automatically use the
oracle defined earlier in the configuration.

******************
 Number of epochs
******************

.. code:: yaml

   number_of_epochs: 1

This parameter defines the number of training epochs.

During each epoch:

#. the experience collector generates episodes;
#. the trainer updates the neural network using the collected data.

***********
 Workspace
***********

.. code:: yaml

   workspace_path: test_workspace

This directory is used to store trained models and temporary data.

In ``HDF5`` mode, intermediate model versions and training datasets are
also written to this directory.

The workspace also stores evaluation artifacts:

-  the current best model at ``oracle.file_path``;
-  saved model checkpoints at ``<oracle_file_path>_<numeric_id>``;
-  the persisted evaluation ratings for those checkpoints at
   ``<oracle_file_path>_scores.yaml``.

.. _alpha_zero_evaluation:

************
 Evaluation
************

.. code:: yaml

   evaluation:
       episodes: 100
       max_old_models: 10
       uncertainty_penalty_coefficient: 3.0
       discount_factor: 1.0
       policy:
           name: DeterministicOraclePolicy
       true_skill:
           mu: 25.0
           sigma: 8.333
           beta: 1.476
           tau: 0.0
           draw_probability: 0.1

This section defines how newly trained models are compared against older
saved models.

The model is updated at every epoch. To prevent performance regression,
each challenger is saved as a separate checkpoint and evaluated against
the other saved checkpoints. The evaluation uses the policy defined in
``policy`` for the lead player and the same policy for the opponents,
with separate oracle instances loaded from the sampled checkpoints.

All checkpoints are rated with TrueSkill using shared evaluation
scenarios. The current best model is the checkpoint with the highest
conservative rating

``mu - uncertainty_penalty_coefficient * sigma``.

The best checkpoint is exposed at ``oracle.file_path``. The underlying
checkpoint files are stored as ``<oracle_file_path>_<numeric_id>``. The
persisted ratings used to resume model selection are stored in
``<oracle_file_path>_scores.yaml``.

-  ``episodes``: number of sampled evaluation scenarios.
-  ``max_old_models``: maximum number of saved checkpoints retained for
   future comparisons.
-  ``uncertainty_penalty_coefficient``: penalty applied to the TrueSkill
   uncertainty ``sigma`` when ranking checkpoints.
-  ``discount_factor``: discount factor used when accumulating observed
   rewards during evaluation games.
-  ``policy``: policy template used for checkpoint evaluation.
-  ``true_skill``: parameters of the TrueSkill rating system used to
   rank checkpoints.

The policy is configurable because model quality is relative to the
policy used to turn the model into actions. A model that performs better
for one policy is not guaranteed to perform better for another. This can
happen not only when comparing direct oracle play with MCTS-based play,
but also when comparing two MCTS policies with different numbers of
simulations.

In practice, the evaluation policy should match the policy you expect to
use when the model is deployed.

************************
 HDF5 specific settings
************************

.. code:: yaml

   hdf5_path_prefix: tic_tac_toe_data
   server_hostname: 127.0.0.1
   server_port: 8888

These settings are only used when the training type is ``HDF5``.

-  ``hdf5_path_prefix``: Prefix used for the HDF5 files created by the
   collection processes.

-  ``server_hostname`` and ``server_port``: The network location where
   the main process runs and accepts data submissions from child
   processes.

*********************
 Standard Input Play
*********************

.. code:: yaml

   stdin_policy:
       name: TicTacToeStdin
       module: mrl.tic_tac_toe.game

This section describes the terminal interface for manual play. It
follows the same structure as ``game_runner``. This section can be
omitted for built-in games. See the :ref:`stdin <game_runner_stdin>` for
details.

**********
 Gui Play
**********

.. code:: yaml

   gui:
       name: make_gui
       module: mrl.tic_tac_toe.tkinter_gui

This section describes the graphical interface for manual play. It
follows the same structure as ``game_runner``. This section can be
omitted for built-in games. See the :ref:`gui <game_runner_gui>` for
details.
