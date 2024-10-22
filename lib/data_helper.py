"""
Various helper functions for working with the data used in this app
"""

import os
import pickle
from glob import glob
from logging import getLogger

logger = getLogger(__name__)


# def pretty_print(env, colors):
#     new_pgn = open("test3.pgn", "at")
#     game = chess.pgn.Game.from_board(env.board)
#     game.headers["Result"] = env.result
#     game.headers["White"], game.headers["Black"] = colors
#     game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
#     new_pgn.write(str(game) + "\n\n")
#     new_pgn.close()
#     pyperclip.copy(env.board.fen())

def find_pgn_files(directory, pattern='*.pgn'):
    dir_pattern = os.path.join(directory, pattern)
    files = list(sorted(glob(dir_pattern)))
    return files

def get_game_data_filenames(flags):
    pattern = os.path.join(flags.play_data_dir, flags.play_data_filename_tmpl % "*")
    files = list(sorted(glob(pattern)))
    return files

def get_next_generation_model_dirs(flags):
    dir_pattern = os.path.join(flags.next_generation_model_dir, flags.next_generation_model_dirname_tmpl % "*")
    dirs = list(sorted(glob(dir_pattern)))
    return dirs

def write_game_data_to_file(path, fname, data):
    if not os.path.isdir(path):
        os.mkdir(path)
    try:
        fpath = os.path.join(path, fname)
        with open(fpath, "wb") as f:
            pickle.dump(obj=data, file=f)
    except Exception as e:
        print(e)

def read_game_data_from_file(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(e)