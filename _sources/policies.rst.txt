########
 Policy
########

A policy is an object that allows a player to choose the next action in
a given state.

Policies must follow the protocol defined below.

.. code:: python

   @runtime_checkable
   class Policy(Protocol[ObservationContra, ActionCo, ActionSpaceContra]):

       @abstractmethod
       def __call__(
           self, observation: ObservationContra, action_space: ActionSpaceContra
       ) -> ActionCo: ...

Below there is a list of policies which are built-in the mrl library.

**************
 RandomPolicy
**************

.. code:: yaml

   name: RandomPolicy

The random policy returns an action selected uniformly at random from a
discrete action space. As a precondition, the action space must be a
sequence.

*******************
 FirstChoicePolicy
*******************

.. code:: yaml

   name: FirstChoicePolicy

The first choice policy always returns the first action listed in the
action space. As a precondition, the action space must be a sequence.

**************
 ManualPolicy
**************

.. code:: yaml

   name: ManualPolicy

This policy represents a human player interacting with the system
through either standard input (terminal) or the GUI. The action is
chosen interactively by the user.

***************************
 DeterministicOraclePolicy
***************************

.. code:: yaml

   name: DeterministicOraclePolicy
   oracle:
       name: RandomRollout
       number_of_rollouts: 15

This policy requires a discrete action space and the definition of an
oracle.

The policy queries the oracle to obtain the probability distribution
over all actions in the action space. It then selects the action with
the highest probability.

************************
 StochasticOraclePolicy
************************

.. code:: yaml

   name: StochasticOraclePolicy
   oracle:
       name: OpenSpielMLP
       capacity:
           input_size: 18
           output_size: 9

This policy requires a discrete action space and the definition of an
oracle.

The policy queries the oracle to obtain a probability distribution over
the available actions. It then selects an action by sampling from this
distribution.

.. _mcts_policy:

************
 MCTSPolicy
************

.. code:: yaml

   name: MCTSPolicy
   oracle:
       name: OpenSpielConv
       capacity:
           input_shape: [2, 7, 7]
           output_size: 7
   mcts:
       number_of_simulations: 10
       pucb_constant: 0.5
       discount_factor: 0.99

This policy requires a discrete action space and an oracle.

The policy performs a Monte Carlo Tree Search (MCTS) starting from the
current state. The simulation is repeated a number of times and uses the
PUCB exploration strategy.

The PUCB policy balances:

-  the expected value of an action (estimated by the oracle), and
-  the number of times the action has already been explored.

Actions that have been explored less frequently are therefore favored
for exploration.

After completing the simulations, the policy selects the action that has
been visited most often during the search.

The parameters have the following meaning:

-  ``number_of_simulations``: number of simulations performed for each
   decision.
-  ``pucb_constant``: exploration weight. Higher values encourage
   exploration of less-visited actions.
-  ``discount_factor``: discount applied to payoffs that occur later in
   the game.

.. _nondet_mcts_policy:

****************************
 NonDeterministicMCTSPolicy
****************************

.. code:: yaml

   name: NonDeterministicMCTSPolicy
   oracle:
       name: RandomRollout
       number_of_rollouts: 15
   mcts:
       number_of_simulations: 10
       pucb_constant: 0.5
       discount_factor: 0.99
       temperature: 0.99

This policy performs the same MCTS simulation as ``MCTSPolicy`` but
selects the final action stochastically.

Instead of choosing the most visited action, the policy samples from the
distribution defined by the visit counts of the actions explored during
the search.

The ``temperature`` parameter controls the amount of randomness:

-  ``temperature = 1.0``: the sampling distribution directly reflects
   the visit frequencies.
-  ``temperature = 0.0``: the policy behaves deterministically and
   selects the most visited action.
-  ``temperature → ∞``: uniform random selection over all actions.
-  intermediate values provide a smooth transition between the extremes.

**********************
 MemoryfullMCTSPolicy
**********************

.. code:: yaml

   name: MemoryfullMCTSPolicy
   oracle:
       name: RandomRollout
       number_of_rollouts: 15
   mcts:
       number_of_simulations: 10
       pucb_constant: 0.5
       discount_factor: 0.99

This policy behaves similarly to ``MCTSPolicy`` but retains the results
of previous simulations.

While ``MCTSPolicy`` discards all simulation data after each call,
``MemoryfullMCTSPolicy`` stores this information. If the same state is
encountered again, the stored statistics are combined with the new
simulations to guide the next decision.

In practice, ``MCTSPolicy`` is often preferred. When the oracle is still
being trained, early simulations may be noisy or random, and retaining
them can negatively affect later decisions.
