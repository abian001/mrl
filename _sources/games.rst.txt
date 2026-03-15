######
 Game
######

A game defines the rules and structure of an interactive environment in
which one or more players make decisions over time.

Conceptually, a game has two main responsibilities:

-  **State generation**: it acts as a factory for creating the initial
   state of a new game.
-  **State transitions**: it defines how the game evolves when players
   take actions, producing new states according to the game rules.

A game also defines **perspectives**. A perspective represents how a
specific player observes the game and which actions are available to
that player. This abstraction allows the framework to support:

-  perfect-information games (all players observe the full state),
-  partial-information games (players observe only part of the state),
-  different observation encodings for different algorithms.

A game must implement the following protocols:

.. code:: python

   @runtime_checkable
   class Game(Protocol[StateCo, Player]):

       @abstractmethod
       def make_initial_state(self) -> StateCo:
           """Create the initial state of the game"""

       @abstractmethod
       def get_players(self) -> tuple[Player, ...]:
           """Returns the player identifiers."""

       @abstractmethod
       def get_perspectives(self) -> Mapping[Player, Perspective]:
           """Return the player perspectives."""


   class Perspective(Protocol[StateContra, ObservationCo, ActionSpaceCo]):

       @abstractmethod
       def get_observation(self, state: StateContra) -> ObservationCo:
           """Returns an observation made by a player under the perspective"""

       @abstractmethod
       def get_action_space(self, state: StateContra) -> ActionSpaceCo:
           """Returns the space of actions the player is allowed to make in the state"""

A game compatible with the AlphaZero algorithm must also implement the
following protocols:

.. code:: python

   @runtime_checkable
   class Restorable(Protocol[StateCo, ObservationContra]):

       @abstractmethod
       def restore(self, observation: ObservationContra) -> StateCo:
           """Returns a state that is compatible with the given observation"""


   @runtime_checkable
   class TurnBased(Protocol[State, ActionContra]):

       @abstractmethod
       def update(self, state: State, action: ActionContra) -> State:
           """Updates the game state according to the action of the active player"""


    class PayoffObservable(Protocol[StateContra, Player, ObservationCo]):

        @abstractmethod
        def get_perspectives(self) -> Mapping[Player, PayoffPerspective[StateContra, ObservationCo]]:
            """Return the player payoff perspectives."""

In addition the perspective for an MCTS game must implemet the following
protocols:

.. code:: python

   @runtime_checkable
   class PayoffPerspective(Perspective, Protocol[StateContra, ObservationCo]):

       @abstractmethod
       def get_payoff(self, state: StateContra) -> float:
           """Returns observed payoff in the given state"""

   @runtime_checkable
   class HasActionSpaceDimension(Protocol):

       @property
       @abstractmethod
       def action_space_dimension(self) -> int:
           """Returns the number of distinct actions allowed in the action space."""

   @runtime_checkable
   class MCTSPerspective(Protocol[StateContra, ActionCo]):

       @abstractmethod
       def get_core(self, state: StateContra) -> np.ndarray:
           """Return the essential part of the observation, used by the oracle
              to compute value and probabilities.
           """

       @abstractmethod
       def get_action_space(self, state: StateContra) -> tuple[ActionCo, ...]:
           """Returns the observed action space as a tuple of actions"""

Note: The signature of ``get_action_space`` in ``MCTSPerspective``
requires the ActionSpace to be a tuple of Action objects, unlike the
more generic ``get_action_space`` signature defined in ``Perspective``.

Below there is a list of games which are built-in the mrl library.

***********
 TicTacToe
***********

.. code:: yaml

   name: TicTacToe
   first_player: X

   name: MCTSTicTacToe
   first_player: random

This is a classic 3×3 grid game in which two players alternately place
their symbol (``X`` or ``O``) on the board. The objective is to create a
line of three identical symbols horizontally, vertically, or diagonally.

Actions are represented by integers from ``0`` to ``8`` corresponding to
the nine cells of the grid.

Two implementations are provided:

-  ``TicTacToe``: a standard implementation suitable for manual play and
   basic policy evaluation.
-  ``MCTSTicTacToe``: an implementation designed for use with MCTS and
   AlphaZero training.

In the MCTS variant, the board state is encoded as a vector of 18
elements by stacking two binary representations: one for the current
player's pieces and one for the opponent's pieces.

The ``first_player`` parameter determines who moves first. It can be
``X``, ``O``, or ``random``.

The game includes built-in terminal and Tk GUI interfaces for manual
play.

***************
 Straight Four
***************

.. code:: yaml

   name: StraightFour
   first_player: O

   name: MCTSStraightFour
   first_player: random

Two players attempt to create a line of four of their symbols on a 7×7
board by dropping tokens into columns.

Actions are represented by integers from ``0`` to ``6`` corresponding to
the seven columns of the board.

Two implementations are provided:

-  ``StraightFour``: a standard implementation suitable for manual play
   and policy evaluation.
-  ``MCTSStraightFour``: a variant adapted for MCTS and AlphaZero
   training.

In the MCTS variant, the board is encoded as two ``7×7`` matrices,
representing the positions of each player's tokens.

The ``first_player`` parameter determines which player moves first
(``X``, ``O``, or ``random``).

Terminal and Tk GUI interfaces are provided for manual play.

*********
 Xiangqi
*********

.. code:: yaml

   name: Xiangqi
   first_player: Red
   step_limit: 200

   name: MCTSXiangqi
   first_player: Black

This is an implementation of **Xiangqi**, a two-player strategy game
played on a 9×10 board. Rules can be found at `xiangqi.com
<https://www.xiangqi.com/how-to-play>`_. Note that repetition rules are
not enforced as in the standard ruleset. In this implementation, the
game always ends in a draw once the step_limit is reached (200 by
default).

Actions are represented as tuples:

``(x_origin, y_origin, x_destination, y_destination)``

which specify the piece to move and its destination square.

Two implementations are available:

-  ``Xiangqi``: a standard version for manual play and evaluation.
-  ``MCTSXiangqi``: a variant designed for MCTS and AlphaZero training.

In the MCTS version, actions are encoded as integers in the range
``0``\ –\ ``127``. The board state is represented as a tensor of shape
``14 × 10 × 9``, where each channel corresponds to the presence of a
specific piece type.

The ``first_player`` parameter determines whether ``Red`` or ``Black``
moves first, although ``Red`` is always the first player in the official
rules.

Terminal and GUI interfaces are provided for manual play.

*********************
 Rock Paper Scissors
*********************

.. code:: yaml

   name: RockPaperScissors
   number_of_rounds: 5

This is an example **discrete-time simultaneous game**.

In each round, both players simultaneously choose one of three actions:
``rock``, ``paper``, or ``scissors``. The winner of the round is
determined by the standard rules:

-  rock beats scissors
-  scissors beats paper
-  paper beats rock

The game is repeated for ``number_of_rounds`` rounds. The overall winner
is the player who wins the most rounds.

Actions are represented by their symbolic names (``rock``, ``paper``,
``scissors``).

This example demonstrates how the framework can represent games where
all players act simultaneously rather than sequentially.
