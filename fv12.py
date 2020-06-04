import numpy as np
from dataclasses import dataclass


from utils.fv import secret_keygen, public_keygen, evaluate_keygen, encrypt, decrypt, add, mul, relin


@dataclass
class Context:
    n: int
    d: int
    T: int
    t: int
    c_q: int
    p_q: np.poly1d


class Fv12:
    def __init__(self, ctx):
        self.ctx = ctx
        self.sk = secret_keygen(ctx.d)
        self.pk = public_keygen(self.sk, ctx.d, ctx.c_q, ctx.p_q)
        self.rlks = evaluate_keygen(self.sk, ctx.d, ctx.T, ctx.c_q, ctx.p_q)

    def encrypt(self, message):
        ciphertext = encrypt(self.pk, message, self.ctx.d, self.ctx.t, self.ctx.c_q, self.ctx.p_q)
        return Ciphertext(ciphertext, self.rlks, self.ctx.T, self.ctx.t, self.ctx.c_q, self.ctx.p_q)

    def decrypt(self, ciphertext):
        return decrypt(self.sk, ciphertext.inner, self.ctx.t, self.ctx.c_q, self.ctx.p_q)


class Ciphertext:
    def __init__(self, inner, rlks, T, t, c_q, p_q):
        self.inner = inner
        self.rlks = rlks
        self.T = T
        self.t = t
        self.c_q = c_q
        self.p_q = p_q

    def __add__(self, other):
        return Ciphertext(
            add(self.inner, other.inner, self.c_q, self.p_q),
            self.rlks,
            self.T,
            self.t,
            self.c_q,
            self.p_q
        )

    def __mul__(self, other):
        result = mul(self.inner, other.inner, self.t, self.c_q, self.p_q)
        return Ciphertext(
            relin(self.rlks, result[0], result[1], result[2], self.T, self.c_q, self.p_q),
            self.rlks,
            self.T,
            self.t,
            self.c_q,
            self.p_q
        )
