import numpy as np
from math import floor, log
from numpy.testing import assert_array_equal


def draw_from_binary(size):
    return np.poly1d((np.random.randint(0, 2, size).astype(int)))


def draw_from_integer(size, coeff_modulus):
    return np.poly1d(np.random.randint(0, coeff_modulus, size).astype(int) % coeff_modulus)


def draw_from_normal(size, coeff_modulus, loc=0, scale=2):
    return np.poly1d(np.random.normal(loc, scale, size).astype(int) % coeff_modulus)


def mod(polynomial, coeff_modulus, poly_modulus):
    return np.poly1d(np.floor(np.polydiv(polynomial, poly_modulus)[1]) % coeff_modulus)


def base_decomp(polynomial, T, coeff_modulus):
    l = floor(log(coeff_modulus, T))
    result = []
    for i in range(l + 1):
        result.append(np.poly1d(np.floor(polynomial / T ** i).astype(int) % T))
    return np.array(result)


# --- Tests ---
def tests():
    n = 4
    d = 2 ** n
    p_q = np.poly1d([1] + ((d - 1) * [0]) + [1])  # x^16 + 1
    # --- Polymod ---
    c_q = 10  # Coefficient modulus
    a = np.poly1d([2] + (14 * [0]))  # 2x^14
    b = np.poly1d([1] + (4 * [0]))  # x^4
    result_mul = a * b  # 2x^14 * x^4 = 2x^18
    assert_array_equal(result_mul, np.poly1d([2] + (18 * [0])))
    result_mod = mod(result_mul, c_q, p_q)  # 2x^18 % x^16 + 1 = -2x^2
    assert_array_equal(result_mod, np.poly1d([8, 0, 0]))
    # --- Base Decomposition ---
    c_q = 2 ** 4
    T = 2  # Decomposition base
    l = floor(log(c_q, T))
    x = np.poly1d([1, 2, 3, 4])
    x_decomposed = base_decomp(x, T, c_q)
    x_reconstructed = np.poly1d(sum(x_decomposed[i] * (T ** i) for i in range(l + 1)))
    assert x_decomposed.shape == (l + 1,)
    assert_array_equal(x_decomposed, np.array([
        np.poly1d([1, 0, 1, 0]),
        np.poly1d([1, 1, 0]),
        np.poly1d([1]),
        np.poly1d([0]),
        np.poly1d([0]),
    ]))
    assert_array_equal(x_reconstructed, x)


if __name__ == '__main__':
    tests()
