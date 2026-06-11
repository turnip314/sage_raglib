from sage.sets.primes import Primes
from sage.symbolic.relation import solve
from sage.arith.misc import gcd
from sage.all import mul, log, floor, binomial, ceil, Ideal, FractionField
from sage.matrix.constructor import Matrix
from sage.symbolic.ring import SR
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import AA, QQbar
from sage.rings.integer import Integer
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.term_order import TermOrder
from sage.rings.continued_fraction import continued_fraction

import math
import random
from itertools import combinations
from fractions import Fraction
from time import time

def extend_ring(R, newvars):
    return PolynomialRing(R, newvars).flattening_morphism().codomain()

def convert_to_ring(R, *args):
    res = []
    for arg in args:
        if type(arg) == list:
            res.append([R(v) for v in arg])
        else:
            res.append(R(arg))

    return tuple(res)

def nextprime(r):
    r = Integer(r)
    return Primes().next(r)

def denom(f):
    return f.denominator()

def numer(f):
    return f.numerator()

def normal(f):
    return f.simplify_rational()

def expand(p):
    # TODO - dummy
    return p
    
def sign(p):
    if p > 0:
        return 1
    elif p < 0:
        return -1
    else:
        return 0
    
def subs(pt, exprs):
    if any(type(k) == list or type(k) == tuple for k in pt.values()):
        pt1 = {v:list(k)[0] for v, k in pt.items()}
        pt2 = {v:list(k)[-1] for v, k in pt.items()}
        if type(exprs) == list:
            return [(expr.subs(pt1), expr.subs(pt2)) for expr in exprs]
        else:
            return (exprs.subs(pt1), exprs.subs(pt2))
    else:
        if type(exprs) == list:
            return [expr.subs(pt) for expr in exprs]
        else:
            return exprs.subs(pt)

def randpoly(vs, degree, dense):
    R = PolynomialRing(ZZ, vs)
    return SR(R.random_element(degree, (1+degree)**len(vs)))

def tdeg(*args):
    return TermOrder('degrevlex', len(args))

def set_precision(a):
    # TODO - we should store everything algebraically so this is not necessary?
    pass

def get_precision():
    # TODO - same as set_precision
    pass

def coeff(p, x, n):
    return p.coefficient(x, n)

def coeffs(f):
    return PolynomialRing(QQ, f.variables())(f).coefficients()

def factors(f):
    fac = PolynomialRing(QQ, f.variables())(f).factor()
    return [fac.unit(), list(fac)]

def primpart(f):
    f = PolynomialRing(QQ, f.variables())(f).coefficients()
    return SR(f/f.content())

def ilog2(v):
    floor(log(v, 2))

def divide(f, g):
    R = PolynomialRing(QQ, list(set(f.variables()) | set(g.variables())))
    f, g = R(f), R(g)
    _, rem = f.quo_rem(g)
    return rem == 0

def combinat_choose(n, k):
    return binomial(n, k)

def degree(expr, vs = None):
    R = PolynomialRing(QQ, list(expr.variables()))
    expr = R(expr)
    if vs is None:
        return expr.degree()
    else:
        vs = [R(v) for v in vs]
        return expr.degree(vs)

def sqrfree(f):
    # TODO - difference from factor?
    fac = f.factor()
    return [fac.unit(), list(fac)]

def Groebner_LeadingMonomial(f):
    return f.lm()

def Groebner_HilbertSeries(gb, vs, newvar):
    R = PolynomialRing(QQ, vs)
    hs = Ideal(gb).change_ring(R).hilbert_series()
    hsubs = {hs.parent().gens()[0]: newvar}
    hs = hs.subs(hsubs)
    hs = FractionField(hs.parent())(hs)
    return hs

def Groebner_NormalForm():
    pass

def LinearAlgebra_RowDimension(M):
    return len(M.rows())

def LinearAlgebra_ColumnDimension(M):
    return len(M.columns())


def linalg_jacobian(Fs, vs):
    J = Matrix(
        [
            [f.derivative(v) for v in vs]
            for f in Fs
        ]
    )
    return J

def isprime(p):
    return p in Primes()

def solve_line(line, vvar):
    if line.degree() > 1:
        debug_exception(line = line, vvar = vvar)
        raise Exception("This should not have been called - check")

    c = line.coefficient(line.parent()(vvar))
    if c == 0:
        debug_exception(line = line, vvar = vvar)
        raise Exception("Cannot solve")
    return -line/c + vvar

def debug(**kwargs):
    print()
    for name, val in kwargs.items():
        print(name + ":", val)
    print()

# For cosmetic purposes
debug_exception = debug