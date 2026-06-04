from sage.sets.primes import Primes
from sage.symbolic.relation import solve
from sage.arith.misc import gcd
from sage.all import mul, log, floor, binomial, ceil
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

def nextprime(r):
    return Primes().next(r)

def denom(f):
    return f.denominator()

def numer(f):
    return f.numerator()

def normal(f):
    return f.simplify_rational()

def expand(p):
    return p.expand()

def indets(exprs):
    if type(exprs == list):
        return list(
            set().union(*[
                v.variables()
                for v in exprs
            ])
        )
    else:
        return exprs.variables()
    
def sign(p):
    if p > 0:
        return 1
    elif p < 0:
        return -1
    else:
        return 0
    
def subs(pt, exprs):
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

def get_precision(a):
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
    fac = PolynomialRing(QQ, f.variables())(f).factor()
    return [fac.unit(), list(fac)]

def Groebner_LeadingMonomial():
    pass

def Groebner_HilbertSeries():
    pass

def Groebner_NormalForm():
    pass

def ElimSaturateIntersect():
    pass

def SaturateIntersect():
    pass

def LinearAlgebra_RowDimension():
    # TODO
    pass
def LinearAlgebra_ColumnDimension():
    # TODO
    pass

def LinearAlgebra_SubMatrix():
    # TODO
    pass

def linalg_jacobian():
    pass

def isprime(p):
    pass