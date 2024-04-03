import numpy as np

BOARD_SIZE = (6,7)
IN_A_ROW = 4
PLAYER_MARKS = [1,2]

horizontal_kernel = np.ones([1, IN_A_ROW], dtype=np.uint8)
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(IN_A_ROW, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)

VICTORY_KERNELS = dict(
    horizontal=horizontal_kernel,
    vertical=vertical_kernel,
    diagonal_identity_matrix=diag1_kernel,
    diagonal_identity_flipped=diag2_kernel,
)