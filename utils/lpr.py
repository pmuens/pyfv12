import numpy as np
from random import choice

from utils.core import draw_from_integer, draw_from_normal, mod


def secret_keygen(d, c_q):
    return draw_from_normal(d, c_q)


def public_keygen(s, d, c_q, p_q):
    a = draw_from_integer(d, c_q)
    e = draw_from_normal(d, c_q)
    pk_0 = mod(-(a * s) + e, c_q, p_q)
    pk_1 = a
    # --- Tests ---
    assert mod(pk_0 + s * pk_1, c_q, p_q) == e
    return pk_0, pk_1


def encrypt(pk, m, d, t, c_q, p_q):
    p_0 = pk[0]
    p_1 = pk[1]
    u = draw_from_normal(d, c_q)
    e_1 = draw_from_normal(d, c_q)
    e_2 = draw_from_normal(d, c_q)
    delta = c_q // t
    c_0 = mod((p_0 * u) + e_1 + (delta * m), c_q, p_q)
    c_1 = mod((p_1 * u) + e_2, c_q, p_q)
    # --- Tests ---
    assert c_q == delta * t + (c_q % t)
    return c_0, c_1


def decrypt(sk, c, t, c_q, p_q):
    scaling = t / c_q
    noisy_plaintext = mod(c[0] + c[1] * sk, c_q, p_q)
    return np.poly1d((np.round(scaling * noisy_plaintext) % t).astype(int))


# --- Tests ---
def tests():
    n = 4
    d = 2 ** n
    t = 7  # Plaintext modulus
    c_q = 1024  # Coefficient modulus
    p_q = np.poly1d([1] + ((d - 1) * [0]) + [1])  # Polynomial modulus --> x^16 + 1
    # --- Secret Key generation ---
    sk = secret_keygen(d, c_q)
    assert len(sk.coeffs) <= d
    # --- Public Key generation ---
    sk = secret_keygen(d, c_q)
    pk = public_keygen(sk, d, c_q, p_q)
    assert len(pk) == 2
    assert len(pk[0].coeffs) <= d
    assert len(pk[1].coeffs) <= d
    # --- Encryption ---
    sk = secret_keygen(d, c_q)
    pk = public_keygen(sk, d, c_q, p_q)
    c = encrypt(pk, 5, d, t, c_q, p_q)
    assert len(c) == 2
    assert len(c[0].coeffs) <= d
    assert len(c[1].coeffs) <= d
    # --- Decryption ---
    sk = secret_keygen(d, c_q)
    pk = public_keygen(sk, d, c_q, p_q)
    m = choice(range(0, 7))
    c = encrypt(pk, m, d, t, c_q, p_q)
    p = decrypt(sk, c, t, c_q, p_q)
    assert p[0] == m


if __name__ == '__main__':
    tests()
