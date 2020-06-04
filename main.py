import sys
import numpy as np

from fv12 import Context, Fv12
from utils.core import tests as tests_utils_core
from utils.lpr import tests as tests_utils_lpr
from utils.fv import tests as tests_utils_fv


def main():
    n = 4
    d = 2 ** n
    T = 2  # Relinearization base
    t = 2 ** 10  # Plaintext modulus
    c_q = (2 ** 20) * t  # Coefficient modulus
    p_q = np.poly1d([1] + ((d - 1) * [0]) + [1])  # Polynomial modulus --> x^16 + 1
    ctx = Context(n, d, T, t, c_q, p_q)
    fv = Fv12(ctx)
    # Addition and multiplication
    m_1 = 10
    m_2 = 20
    m_3 = 3
    c_1 = fv.encrypt(m_1)
    c_2 = fv.encrypt(m_2)
    c_3 = fv.encrypt(m_3)
    res = fv.decrypt(((c_1 + c_2) * c_3))
    print(f'({m_1} + {m_2}) * {m_3} = {res[0]}')
    assert res[0] == (m_1 + m_2) * m_3


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    if len(sys.argv) > 1 and 'test' in sys.argv[1]:
        tests_utils_core()
        tests_utils_lpr()
        tests_utils_fv()
        print('Success: 3/3')
