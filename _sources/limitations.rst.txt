#############
 Limitations
#############

*********************
 Training Visibility
*********************

TicTacToe is a very simple game, and an optimal strategy can be computed
quickly using algorithms other than AlphaZero (for example, a solver for
Nash equilibria). The optimal strategy has a 0% loss rate. I have not
performed exhaustive testing, so it is possible that the parameter
choices used here are not optimal. At present, however, it is difficult
to determine whether a change in parameters actually improves training.

The MRL framework provides limited visibility into the training process.
The only feedback available is a periodic evaluation against a random
policy, which offers little insight into what happens during training.
In particular, there is no information about:

-  How well each training round fits the generated training examples.
-  How changes in the training dataset affect the model’s behaviour over
   time.

For example, the framework does not allow detection of situations where
the model cycles between parameter configurations. Because training uses
only the most recent self-play examples, earlier mistakes and successes
are forgotten. This can cause the model to repeatedly relearn the same
lessons, effectively entering a training loop.

Additionally, there is currently no mechanism to run automated
hyperparameter searches, where the discovery of effective parameter
configurations happens automatically.

****************
 Training Speed
****************

The current implementation of data collection and training setup is not
optimized for performance. Most development effort was focused on
flexibility and modularity, with relatively little attention paid to
reducing computational overhead.

Profiling TicTacToe training for two epochs shows that PyTorch neural
network computations account for approximately 55% of the total runtime,
while Monte Carlo Tree Search (MCTS) simulations account for about 35%.

However, the largest computational bottleneck appears when updating the
state of more complex games. For example, profiling Xiangqi training for
two epochs shows that neural network computation accounts for only about
11% of the runtime, whereas updating the game states accounts for
roughly 55%. In this case, performance could likely be improved by
adopting matrix-based representations similar to those used in engines
such as Stockfish.

Due to the slow training speed, meaningful results were not obtained for
Xiangqi after 1–2 hours of training. Substantially longer training runs
are therefore likely necessary.

************
 Interfaces
************

The gameplay interfaces currently have poor usability. Their primary
goal was to provide minimal tools for quickly testing trained policies,
rather than offering a polished user experience.

As a result, interaction can be confusing. For example, in the Xiangqi
interface, when a user makes an invalid move nothing happens, which
makes it difficult to understand that the move was rejected.
