# MCTS Connect Four
 
https://github.com/plkmo/AlphaZero_Connect4/blob/master/src/

https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning

kaggle competitions submit -c connectx -f submission.tar.gz -m "MCTS 75"
tar --exclude=*.pyc --exclude=*__pycache__ -czvf submission.tar.gz agent c4_game c4_gym config.yaml mcts_phase4.pt main.py utility_constants.py