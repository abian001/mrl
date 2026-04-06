###############
 Release notes
###############

************
 MRL v0.3.0
************

-  Added a new predefined policy, AlphaBetaPolicy, implementing the
   alpha-beta search algorithm with rollouts.

-  Made report generator policies configurable, enabling monitoring of
   the trained policy against user-defined combinations of test
   policies.

-  Replaced the model retention scheme based on direct comparison with
   previous models with a tournament-based ranking system using the
   TrueSkill rating algorithm.

-  Standardized terminology: reward now refers to immediate game
   transition outcomes, while payoff denotes the outcome of an entire
   game or play sequence.

-  Renamed PayoffPerspective to RewardPerspective and PayoffObservable
   to RewardObservable.

************
 MRL v0.2.0
************

-  Fixed issues that were negatively affecting training effectiveness
-  Added support for Dirichlet root noise in MCTS simulations to
   increase training data diversity
-  Made the evaluation policy configurable, enabling optimization of
   trained models for specific evaluation strategies
-  Fixed an issue that prevented training from resuming correctly after
   an initial session
-  Improved validation messages for incorrect configurations

************
 MRL v0.1.0
************

The initial release v0.1.0 includes:

-  The game framework and the game runner;
-  An implementation of AlphaZero;
-  The implementation of example games: TicTacToe, StraightFour and
   Xiangqi;
-  Documentation, tutorials and examples.
