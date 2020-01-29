import numpy as np
q_preay = np.array(np.array([# Left, Right, Forward // # L_recent, T, O, L, RF
            np.array([0, 0, 1]),  # 00000
            np.array([1, 0, 0]),  # 00001
            np.array([0, 1, 0]),  # 00010
            np.array([1, 0, 0]),  # 00011
            np.array([1, 0, 0]),  # 00100
            np.array([1, 0, 0]),  # 00101
            np.array([0, 1, 0]),  # 00110
            np.array([1, 0, 0]),  # 00111

            np.array([0, 0, 1]),  # 01000
            np.array([1, 0, 0]),  # 01001
            np.array([0, 1, 0]),  # 01010
            np.array([1, 0, 0]),  # 01011
            np.array([1, 0, 0]),  # 01100
            np.array([1, 0, 0]),  # 01101
            np.array([0, 1, 0]),  # 01110
            np.array([1, 0, 0]),  # 01111

            np.array([0, 0, 1]),  # 10000
            np.array([1, 0, 0]),  # 10001
            np.array([0, 1, 0]),  # 10010
            np.array([1, 0, 0]),  # 10011
            np.array([0, 1, 0]),  # 10100
            np.array([1, 0, 0]),  # 10101
            np.array([0, 1, 0]),  # 10110
            np.array([1, 0, 0]),  # 10111

            np.array([0, 0, 1]),  # 11000
            np.array([1, 0, 0]),  # 11001
            np.array([0, 1, 0]),  # 11010
            np.array([1, 0, 0]),  # 11011
            np.array([0, 1, 0]),  # 11100
            np.array([1, 0, 0]),  # 11101
            np.array([0, 1, 0]),  # 11110
            np.array([1, 0, 0]),  # 11111
        ]))
