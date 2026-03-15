########
 Oracle
########

An oracle is an object that estimates the expected payoff for a player
in a given state and determines the optimal strategy for that state. A
strategy is represented as a probability distribution over the actions
available to the player.

An oracle must implement the following protocol:

.. code:: python

   class Oracle(ABC, Generic[Observation]):

       @abstractmethod
       def get_value(self, observation: Observation) -> Payoff:
           """Return the expected payoff in the given observation"""

       @abstractmethod
       def get_probabilities(
           self, observation: Observation, legal_mask: LegalMask
       ) -> Probabilities:
           """Returns the probabilities for each action.
           The legal mask defines which actions are allowed.
           """

Below there is a list of oracles which are built-in the mrl library.

***************
 RandomRollout
***************

.. code:: yaml

   name: RandomRollout
   number_of_rollouts: 15

This oracle estimates the expected payoff by performing
``number_of_rollouts`` simulations using random actions.

The expected payoff is computed as the average payoff across all
simulations. The returned action probability distribution is uniform
over the legal actions.

**************
 OpenSpielMLP
**************

.. code:: yaml

   name: OpenSpielMLP
   capacity:
       input_size: 18
       output_size: 9
       nn_width: 2
       nn_depth: 2

This oracle is implemented as a multi-layer perceptron (MLP). It takes
as input the state of the game encoded as an ``n``-dimensional vector
and produces:

-  a probability distribution over the available actions, and
-  an estimate of the expected payoff.

The neural network follows the `OpenSpiel
<https://github.com/google-deepmind/open_spiel>`_ architecture. It
contains ``nn_depth`` hidden layers in addition to the input and output
layers. Each hidden layer has ``nn_width`` neurons, and each layer is
fully connected to the next.

The output consists of two heads:

-  a **policy head**, which produces logits for the action probability
   distribution;
-  a **value head**, which estimates the expected payoff.

``input_size`` defines the dimension of the input vector, while
``output_size`` defines the number of possible actions and therefore the
dimension of the probability distribution.

***************
 OpenSpielConv
***************

.. code:: yaml

   name: OpenSpielConv
   capacity:
       input_shape: [2, 7, 7]
       output_size: 7
       nn_width: 2
       nn_depth: 2

This oracle is implemented as a convolutional neural network. It takes
as input the state of the game encoded as a array with shape ``(channels
× height × width)``. Reshaping is performed automatically, so a
perspective may return an array of any shape, provided that the total
number of elements equals channels × height × width.

The network produces both:

-  a probability distribution over the available actions, and
-  an estimate of the expected payoff.

The architecture follows the `OpenSpiel
<https://github.com/google-deepmind/open_spiel>`_ design. It includes
``nn_depth`` convolutional layers in addition to the input and output
layers. Each convolutional layer produces ``nn_width`` output channels.

Batch normalization and ReLU activation functions are applied after each
convolutional layer.

*****************
 OpenSpielResnet
*****************

.. code:: yaml

   name: OpenSpielResnet
   capacity:
       input_shape: [2, 7, 7]
       output_size: 7
       nn_width: 2
       nn_depth: 2

This oracle is implemented as a convolutional neural network with
residual connections.

The network takes the game state encoded as a tensor with shape
``(channels × height × width)`` and produces both a probability
distribution over the available actions and an estimate of the expected
payoff.

The architecture follows the `OpenSpiel
<https://github.com/google-deepmind/open_spiel>`_ residual design. It
includes ``nn_depth`` residual blocks in addition to the input and
output layers.

Each residual block contains two convolutional layers producing
``nn_width`` channels. The output of a residual block is obtained by
adding the block input to the output of the second convolutional layer,
forming a residual connection.
