predefined_stdin_policies = {
    'TicTacToe': ('TicTacToeStdin', 'mrl.tic_tac_toe.game'),
    'MCTSTicTacToe': ('TicTacToeStdin', 'mrl.tic_tac_toe.game'),
    'StraightFour': ('StraightFourStdin', 'mrl.straight_four.game'),
    'MCTSStraightFour': ('StraightFourStdin', 'mrl.straight_four.game'),
    'Xiangqi': ('XiangqiStdin', 'mrl.xiangqi.game'),
    'MCTSXiangqi': ('XiangqiStdin', 'mrl.xiangqi.enumerable_game'),
    'RockPaperScissors': ('RockPaperScissorsStdin', 'mrl.rock_paper_scissors.game')
}

predefined_modules = {
    'TicTacToe': 'mrl.tic_tac_toe.game',
    'MCTSTicTacToe': 'mrl.tic_tac_toe.mcts_game',
    'StraightFour': 'mrl.straight_four.game',
    'MCTSStraightFour': 'mrl.straight_four.mcts_game',
    'Xiangqi': 'mrl.xiangqi.game',
    'MCTSXiangqi': 'mrl.xiangqi.mcts_game',
    'RockPaperScissors': 'mrl.rock_paper_scissors.game',
    'RandomPolicy': 'mrl.test_utils.policies',
    'FirstChoicePolicy': 'mrl.test_utils.policies',
    'ManualPolicy': 'mrl.test_utils.policies',
    'AlphaBetaPolicy': 'mrl.test_utils.policies',
    'OpenSpielMLP': 'mrl.alpha_zero.models',
    'OpenSpielConv': 'mrl.alpha_zero.models',
    'OpenSpielResnet': 'mrl.alpha_zero.models',
    'RandomRollout': 'mrl.alpha_zero.random_rollout',
    'DeterministicOraclePolicy': 'mrl.alpha_zero.oracle',
    'StochasticOraclePolicy': 'mrl.alpha_zero.oracle',
    'MCTSPolicy': 'mrl.alpha_zero.mcts',
    'NonDeterministicMCTSPolicy': 'mrl.alpha_zero.mcts',
    'MemoryfullMCTSPolicy': 'mrl.alpha_zero.mcts'
} | {
    name: module
    for (_, (name, module)) in predefined_stdin_policies.items()
}

predefined_guis = {
    'TicTacToe': ('make_gui', 'mrl.tic_tac_toe.tkinter_gui'),
    'MCTSTicTacToe': ('make_gui', 'mrl.tic_tac_toe.tkinter_gui'),
    'StraightFour': ('make_gui', 'mrl.straight_four.tkinter_gui'),
    'MCTSStraightFour': ('make_gui', 'mrl.straight_four.tkinter_gui'),
    'Xiangqi': ('make_gui', 'mrl.xiangqi.tkinter_gui'),
    'MCTSXiangqi': ('make_gui', 'mrl.xiangqi.tkinter_gui')
}
