###################
 Quick Start Guide
###################

This quick start guide will help you **play a game and run a minimal
AlphaZero training loop**. By the end, you should be able to run a game
in the terminal, see the output, and start experimenting.

*****************
 1. Installation
*****************

You can run the library using **Docker** (recommended) or a local Python
environment.

**Using Docker:**

.. code:: bash

   # Build the production image
   docker compose build mrl_prod

   # Start a container interactively
   docker compose run --rm mrl_prod

**Local Python environment (optional):**

Install dependencies:

.. code:: bash

   python -m venv venv
   source venv/bin/activate
   pip install torch==2.8.0 pyyaml==6.0.3 h5py==3.15.1 pydantic==2.12.4

Install MRL:

.. code:: bash

   pip install .

Test that the library is available:

.. code:: bash

   run_game -h

**********************
 2. Run a simple game
**********************

We will start with **TicTacToe** and play against a random policy in the
terminal.

.. code:: bash

   run_game examples/tic_tac_toe_manual.yaml --mode terminal

You should see a 3x3 grid and be prompted to make moves. Press the keys
corresponding to the cell you want to place your symbol in.

****************************
 3. Evaluate a policy match
****************************

Run a game automatically and see statistics for policy performance:

.. code:: bash

   run_game examples/tic_tac_toe_auto.yaml --mode evaluate

This will run multiple simulations and show how the players perform.

You will a report similar to this one.

.. code:: bash

   Total plays as player O: N. 100
   Mean Payoff: 0.665
   Payoff distribution in buckets:
   (-inf, 0.25): 28 (28%)
   (0.25, 0.75): 11 (11%)
   (0.75, inf): 61 (61%)

The report indicates that 100 games were simulated. Player O achieved a
mean payoff of 0.665. In Tic-Tac-Toe, the payoff is 0 for losses, 1 for
wins, and 0.5 for draws. Accordingly, the three buckets above represent
losses, draws, and wins, respectively.

******************************
 4. Train a minimal AlphaZero
******************************

Run a **smoke test** of AlphaZero training:

.. code:: bash

   run_alpha_zero examples/tic_tac_toe_alpha_zero.yaml --mode train

This will perform a few self-play episodes, collect experiences using
the NonDeterministicMCTSPolicy, and update a neural network oracle.

***********************************
 5. Play against the trained model
***********************************

Play against the model you just trained in the terminal:

.. code:: bash

   run_alpha_zero examples/tic_tac_toe_alpha_zero.yaml --mode terminal

************************
 6. Optional next steps
************************

Once you have succeeded with the minimal workflow, you can explore:

-  **Change games:** Try `StraightFour`, `Xiangqi`, or
   `RockPaperScissors`.
-  **Experiment with policies:** Use MCTS, deterministic, or stochastic
   oracle policies.
-  **Use GUI:** Replace `--mode terminal` with `--mode gui` to use the
   built-in Tkinter GUI.
-  **Modify AlphaZero parameters:** Edit training YAML files to increase
   the number of simulations, episodes, or epochs.

**************************
 7. Architecture in brief
**************************

The library is built around a few core abstractions:

-  **Game:** Produces states and enforces rules.
-  **State:** Can be any structure, must include `is_final` (and
   `active_player` for turn-based games).
-  **Perspective:** Defines what each player sees and optionally
   provides `get_payoff(state)`.
-  **Policy:** Chooses actions based on observations and action spaces.
-  **Oracle:** Evaluates states and provides action probabilities (used
   by MCTS and AlphaZero).

This separation allows **new games, policies, and neural networks** to
be plugged in easily.

********************
 8. Troubleshooting
********************

-  If you cannot run Docker GUI apps on Mac, make sure `XQuartz
   <https://www.xquartz.org/>`_ is installed.
-  Use `run_game -h` or `run_alpha_zero -h` to see all command line
   options.
