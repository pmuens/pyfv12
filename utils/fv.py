import numpy as np
from math import floor, log
from random import choice

from utils.core import draw_from_binary, draw_from_integer, draw_from_normal, mod, base_decomp
from utils.lpr import public_keygen as lpr_public_keygen
from utils.lpr import decrypt as lpr_decrypt


def secret_keygen(d):
    return draw_from_binary(d)


def public_keygen(s, d, c_q, p_q):
    return lpr_public_keygen(s, d, c_q, p_q)


# NOTE: We're implementing Relinearization Version 1 here
def evaluate_keygen(s, d, T, c_q, p_q):
    l = floor(log(c_q, T))
    rlks = []
    # NOTE: [0..l] is inclusive (see: https://en.wikipedia.org/wiki/Interval_(mathematics)#Integer_intervals)
    for i in range(l + 1):
        a_i = draw_from_integer(d, c_q)
        e_i = draw_from_normal(d, c_q)
        rlk_0 = mod(-(a_i * s) + e_i + ((T ** i) * (s ** 2)), c_q, p_q)
        rlk_1 = a_i
        # --- Tests ---
        assert mod(rlk_0 + (rlk_1 * s - (T ** i) * (s ** 2)), c_q, p_q) == e_i
        rlks.append((rlk_0, rlk_1))
    return rlks


def encrypt(pk, m, d, t, c_q, p_q):
    p_0 = pk[0]
    p_1 = pk[1]
    u = draw_from_binary(d)
    e_1 = draw_from_normal(d, c_q)
    e_2 = draw_from_normal(d, c_q)
    delta = c_q // t
    c_0 = mod((p_0 * u) + e_1 + (delta * m), c_q, p_q)
    c_1 = mod((p_1 * u) + e_2, c_q, p_q)
    # --- Tests ---
    assert c_q == delta * t + (c_q % t)
    return c_0, c_1


def decrypt(sk, c, t, c_q, p_q):
    return lpr_decrypt(sk, c, t, c_q, p_q)


def add(c_1, c_2, c_q, p_q):
    c_res_0 = mod(c_1[0] + c_2[0], c_q, p_q)
    c_res_1 = mod(c_1[1] + c_2[1], c_q, p_q)
    return c_res_0, c_res_1


def mul(c_1, c_2, t, c_q, p_q):
    scaling = t / c_q
    noisy_c_0_numerator = c_1[0] * c_2[0]
    c_res_0 = mod(np.round(scaling * noisy_c_0_numerator), c_q, p_q)
    noisy_c_1_numerator = (c_1[0] * c_2[1]) + (c_1[1] * c_2[0])
    c_res_1 = mod(np.round(scaling * noisy_c_1_numerator), c_q, p_q)
    noisy_c_2_numerator = c_1[1] * c_2[1]
    c_res_2 = mod(np.round(scaling * noisy_c_2_numerator), c_q, p_q)
    return c_res_0, c_res_1, c_res_2


def relin(rlks, c_0, c_1, c_2, T, c_q, p_q):
    l = floor(log(c_q, T))
    c_2_decomp = base_decomp(c_2, T, c_q)
    c_res_0 = mod(c_0 + sum(rlks[i][0] * c_2_decomp[i] for i in range(l + 1)), c_q, p_q)
    c_res_1 = mod(c_1 + sum(rlks[i][1] * c_2_decomp[i] for i in range(l + 1)), c_q, p_q)
    # --- Tests ---
    assert len(rlks) == len(c_2_decomp)
    return c_res_0, c_res_1


# --- Tests ---
def tests():
    n = 4
    d = 2 ** n
    T = 2  # Relinearization base
    t = 2 ** 10  # Plaintext modulus
    c_q = (2 ** 20) * t  # Coefficient modulus
    p_q = np.poly1d([1] + ((d - 1) * [0]) + [1])  # Polynomial modulus --> x^16 + 1
    # --- Secret Key generation ---
    sk = secret_keygen(d)
    assert len(sk.coeffs) <= d
    # --- Public Key generation ---
    sk = secret_keygen(d)
    pk = public_keygen(sk, d, c_q, p_q)
    assert len(pk) == 2
    assert len(pk[0].coeffs) <= d
    assert len(pk[1].coeffs) <= d
    # --- Evaluate Key generation ---
    sk = secret_keygen(d)
    rlks = evaluate_keygen(sk, d, T, c_q, p_q)
    assert len(rlks) == floor(log(c_q, T)) + 1
    # --- Encryption ---
    sk = secret_keygen(d)
    pk = public_keygen(sk, d, c_q, p_q)
    c = encrypt(pk, 3, d, t, c_q, p_q)
    assert len(c) == 2
    assert len(c[0].coeffs) <= d
    assert len(c[1].coeffs) <= d
    # --- Decryption ---
    sk = secret_keygen(d)
    pk = public_keygen(sk, d, c_q, p_q)
    m = choice(range(0, 7))
    c = encrypt(pk, m, d, t, c_q, p_q)
    p = decrypt(sk, c, t, c_q, p_q)
    assert p[0] == m
    # --- Add ---
    sk = secret_keygen(d)
    pk = public_keygen(sk, d, c_q, p_q)
    m_1 = 2
    m_2 = 3
    c_1 = encrypt(pk, m_1, d, t, c_q, p_q)
    c_2 = encrypt(pk, m_2, d, t, c_q, p_q)
    res = add(c_1, c_2, c_q, p_q)
    p = decrypt(sk, res, t, c_q, p_q)
    assert p[0] == (m_1 + m_2) % t
    # --- Mul ---
    sk = secret_keygen(d)
    pk = public_keygen(sk, d, c_q, p_q)
    rlks = evaluate_keygen(sk, d, T, c_q, p_q)
    m_1 = 2
    m_2 = 3
    c_1 = encrypt(pk, m_1, d, t, c_q, p_q)
    c_2 = encrypt(pk, m_2, d, t, c_q, p_q)
    res = mul(c_1, c_2, t, c_q, p_q)
    # Test result without relinearization
    p = np.poly1d(np.round(t / c_q * mod(res[0] + res[1] * sk + res[2] * sk ** 2, c_q, p_q)) % t)
    assert p[0] == (m_1 * m_2) % t
    # --- Relin ---
    # Test result with relinearization
    res = relin(rlks, res[0], res[1], res[2], T, c_q, p_q)
    p = decrypt(sk, res, t, c_q, p_q)
    assert p[0] == (m_1 * m_2) % t


if __name__ == '__main__':
    tests()
