#############
 Game runner
#############

The game runner is a command-line utility used to execute games. It can
be used to:

-  evaluate the outcome of policies playing against each other across
   multiple simulations;
-  play the game interactively against a policy, provided that an
   appropriate interface is implemented.

The game runner can be invoked from the command line as follows:

.. code:: bash

   run_game examples/tic_tac_toe_auto.yaml --mode evaluate

There are three available modes:

-  ``evaluate``: evaluate policies against each other over multiple
   runs.
-  ``terminal``: play against a predefined policy using a standard-input
   interface (if implemented).
-  ``gui``: play against a predefined policy using a graphical interface
   (if implemented).

***********************
 Example configuration
***********************

.. code:: yaml

   game:
       name: 'Coordination'
       module: 'coordination'
       number_of_tests: 5
   policies:
       O:
           name: policy_1
       X:
           name: 'MCTSPolicy'
           oracle: oracle_1
           mcts:
               number_of_simulations: 125
   shared_policies:
       policy_1:
           name: 'MCTSPolicy'
           oracle:
               name: 'RandomRollout'
               number_of_rollouts: 15
           mcts:
               number_of_simulations: 125
   oracles:
       oracle_1:
           name: 'RandomRollout'
           number_of_rollouts: 10
   evaluation:
       observed_players: ['O']
       number_of_tests: 10
       buckets:
           - [-inf, 0.50]
           - [0.50, +inf]
   stdin_policy:
       name: CoordinationStdin
       module: coordination
   gui:
       name: CoordinationGui
       module: coordination_gui

The YAML configuration file is structured into several sections.

.. _game_runner_game:

******
 Game
******

.. code:: yaml

   game:
       name: 'Coordination'
       module: 'coordination'
       number_of_tests: 5

The ``game`` section contains the information required to construct the
game instance.

-  ``name``: the name of the game class.
-  ``module``: the module path where the class is defined.
-  additional parameters correspond to arguments of the game class
   constructor.

Predefined games included in the library do not require the ``module``
field.

**********
 Policies
**********

.. code:: yaml

   policies:
       O:
           name: policy_1
       X:
           name: 'MCTSPolicy'
           oracle: oracle_1
           mcts:
               number_of_simulations: 125

The ``policies`` section defines the policy used by each player.

Each entry must specify the policy name and optionally a module. If the
policy is part of the built-in library, the module does not need to be
specified.

Policies can be defined in two ways:

-  inline, directly inside the player configuration;
-  by referencing a policy defined in the ``shared_policies`` section.

Some policies require an oracle. In that case, the oracle can either be
defined inline or referenced from the ``oracles`` section.

*****************
 Shared policies
*****************

.. code:: yaml

   shared_policies:
       policy_1:
           name: 'MCTSPolicy'
           oracle:
               name: 'RandomRollout'
               number_of_rollouts: 15
           mcts:
               number_of_simulations: 125

The ``shared_policies`` section defines policy configurations that are
used by multiple players. This helps avoid duplication when several
players use the same policy setup.

****************
 Shared oracles
****************

.. code:: yaml

   oracles:
       oracle_1:
           name: 'RandomRollout'
           number_of_rollouts: 10

The ``oracles`` section defines oracle configurations that can be shared
across multiple policies.

******************
 Evaluation setup
******************

.. code:: yaml

   evaluation:
       observed_players: ['O']
       number_of_tests: 10
       buckets:
           - [-inf, 0.50]
           - [0.50, +inf]

The ``evaluation`` section specifies how the evaluation is performed and
how the results are summarized.

-  ``number_of_tests``: number of simulations to run.
-  ``observed_players``: list of players whose payoffs will be analyzed.
-  ``buckets``: payoff ranges used to group results.

At the end of the evaluation, the payoff values are grouped into the
defined buckets. The final report shows how many runs produced payoffs
within each bucket.

.. _game_runner_stdin:

*********************
 Standard Input Play
*********************

.. code:: yaml

   stdin_policy:
       name: CoordinationStdin
       module: coordination

The ``stdin_policy`` section defines the policy used for terminal-based
interaction.

This must be a policy derived from ``InteractivePolicy``. See the
Coordination game example for a reference implementation.

There is no need to define this section when using the built-in
standard-input interface for games already included in the library.

.. _game_runner_gui:

**********
 Gui Play
**********

.. code:: yaml

   gui:
       name: make_gui
       module: coordination_gui

The ``gui`` section defines the graphical interface used when running
the game in GUI mode.

The specified name must refer to a constructor function that returns an
``mrl.tkinter_gui.gui.Gui`` object. Examples can be found in the GUI
implementations for Tic Tac Toe and Xiangqi.

As with the terminal interface, this section is not required for games
that already include a built-in GUI.
