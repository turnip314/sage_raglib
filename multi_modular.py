# This file is part of RAGlib (Real Algebraic Geometry Library).
#
# RAGlib is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# RAGlib is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RAGlib.  If not, see <https://www.gnu.org/licenses/>
#
# Authors:
# Mohab Safey El Din

# Multi-modular routines used in RAGlib
# Most of them should be implemented in msolve in a not too far
# future (hopefully)

from sage.all import (
    Integer, Set, ZZ, QQ,
    PolynomialRing, GF,
    next_prime,
    crt,           # replaces Maple's chrem (Chinese Remainder Theorem)
    rational_reconstruction,  # replaces Maple's iratrecon
    prod,
    randint,
    Ideal
)
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.polynomial.term_order import TermOrder
import sys
from msolve import MSolveGroebner, MSolveGroebnerLM


# ---------------------------------------------------------------------------
# Helper: leading monomial of a polynomial under grevlex order over vars
# ---------------------------------------------------------------------------

def _leading_monomial(pol, vars):
    """Return the leading monomial of pol under grevlex (tdeg) order."""
    R = pol.parent()
    return R(pol.lm())


# ---------------------------------------------------------------------------

def IsMinimal(candidates, lm, fc, vars, opts=None):
    if opts is None:
        opts = {}
    newlm = MSolveGroebnerLM(candidates, fc, vars, opts)
    newlm = set(newlm)
    return newlm == lm


def MinimalGeneratorsDichotomy(gb, sys, a, b, lm, fc, vars, opts=None):
    if opts is None:
        opts = {}
    if b - a <= 1:
        boo = IsMinimal(list(gb[:b]) + list(sys), lm, fc, vars, opts)
        if boo:
            return b
        else:
            return MinimalGeneratorsDichotomy(gb, sys, b, b + 1, lm, fc, vars, opts)
    N = (a + b) // 2   # Maple's iquo is integer division

    boo = IsMinimal(list(gb[:N]) + list(sys), lm, fc, vars, opts)
    if boo:
        return MinimalGeneratorsDichotomy(gb, sys, a, N, lm, fc, vars, opts)
    else:
        return MinimalGeneratorsDichotomy(gb, sys, N, b, lm, fc, vars, opts)


# gb is a GB for the grevlex order over vars, fc is the characteristic
# gb is ordered increasingly
# sys is a list of extra polynomials known to belong to the ideal
# returns the number of elements of gb needed to generate the same ideal
# as gb starting from the first element of gb
def MinimalGenerators(gb, sys, vars, fc, opts=None):
    if opts is None:
        opts = {}
    lm = set(_leading_monomial(g, vars) for g in gb)
    return MinimalGeneratorsDichotomy(gb, sys, 1, len(gb), lm, fc, vars, opts)


##############################################################################
##############################################################################

# newfc is a prime
# witnessmod is a list of witness coefficients (taken modulo modulus)
# newvaluesmod is a list of witness coefficients (taken modulo newfc)
def WitnessLift(newfc, witnessmod, newvaluesmod, modulus):
    newmodulus = modulus * newfc

    # chrem (Chinese Remainder Theorem): lift each coefficient pair
    newwitnessmod = [
        crt(Integer(w), Integer(v), Integer(modulus), Integer(newfc))
        for w, v in zip(witnessmod, newvaluesmod)
    ]

    witness = []
    for i in range(len(witnessmod)):
        # iratrecon: rational reconstruction; returns None/FAIL when it cannot reconstruct
        u = rational_reconstruction(newwitnessmod[i], newmodulus)
        if u is None:
            return [], newwitnessmod, newmodulus
        else:
            witness.append(u)
    return witness, newwitnessmod, newmodulus


# sys is a list of polynomials
# lsupport is a list of monomial supports
# vars is the list of variables
#
# returns boo, set such that boo=False and set = newsupport if some element in
# sys has a support larger than the corresponding element in lsupport, otherwise
# it returns True, {}
def MonomialSupport(sys, lsupport, vars):
    R = sys[0].parent() if sys else None
    newlsupport = []
    boo = True
    for i in range(len(sys)):
        pol = sys[i]
        # coeffs(..., vars, 'm') in Maple gives coefficients and stores monomials in 'm'
        # In SageMath: pol.monomials() gives the list of monomials
        m = set(pol.monomials())
        if m <= set(lsupport[i]):
            newlsupport.append(lsupport[i])
        else:
            # Sort monomials by grevlex order (ascending, i.e. smallest first)
            sorted_m = sorted(list(m), key=lambda mono: R(mono).lm(), reverse=False)
            newlsupport.append(sorted_m)
            boo = False
    if not boo:
        return False, newlsupport
    return True, {}


# pol is a polynomial
# vars is the list of variables
# support is a list of monomials containing the support of pol
# support is assumed to be ordered increasingly by grevlex(vars)
# returns the list of corresponding coefficients
def TableCoeffsSinglePoly(pol, vars, support):
    # In SageMath, coefficients() and monomials() return parallel lists
    lm = pol.monomials()
    lc_map = {mono: coeff for coeff, mono in zip(pol.coefficients(), pol.monomials())}

    # Sort monomials by grevlex (ascending)
    R = pol.parent()
    lm_sorted = sorted(lm, key=lambda mono: R(mono).lm(), reverse=False)

    lctable = [ZZ(0)] * len(support)
    for mono in lm_sorted:
        if mono in support:
            k = support.index(mono)
            lctable[k] = lc_map[mono]
    return lctable


def OldTableCoeffs(sys, vars, lsupport, lctables):
    newlctables = [[] for _ in range(len(sys))]
    for i in range(len(sys)):
        pol = sys[i]
        lc = TableCoeffsSinglePoly(pol, vars, lsupport[i])
        newlctables[i] = [list(lctables[i][k]) + [lc[k]] for k in range(len(lc))]
    return newlctables


def TableCoeffs(sys, vars, lsupport, lctables):
    newlctables = []
    for i in range(len(sys)):
        pol = sys[i]
        lc = TableCoeffsSinglePoly(pol, vars, lsupport[i])
        newlctables.append([list(lctables[i][k]) + [lc[k]] for k in range(len(lc))])
    return newlctables


def LiftPolynomials(lctables, support, primetable, modulus, islifted):
    lifted = []
    for i in range(len(lctables)):
        pol = 0
        if i >= islifted:   # Maple: i > islifted (1-indexed), Python: i >= islifted (0-indexed)
            length = len(lctables[i])
            for j in range(len(lctables[i])):
                cc = crt(lctables[i][j], primetable)   # chrem over list of residues/moduli
                cc = rational_reconstruction(cc, modulus)
                if cc is None:
                    return lifted
                # support[i] is stored ascending; reverse index mirrors Maple's length-j+1
                pol = pol + cc * support[i][length - j - 1]
            lifted.append(pol)
    return lifted


def NewValuesWitness(newsys, vars, support):
    newvaluesmod = []
    for i in range(len(newsys)):
        pol = newsys[i]
        # Get monomials and their coefficient map
        lc_map = {mono: coeff for coeff, mono in zip(pol.coefficients(), pol.monomials())}
        m = list(pol.monomials())
        # Sort ascending by grevlex
        R = pol.parent()
        m_sorted = sorted(m, key=lambda mono: R(mono).lm(), reverse=False)
        # support[i][-1] is the largest monomial (last in ascending order)
        target = support[i][-1]
        lc_val = lc_map.get(target, 0)
        newvaluesmod.append(lc_val)
    return newvaluesmod


def GeneratorsLift(newsys, newfc, vars, modulus, witnessmod, support,
                   systable, primetable):
    newsystable = list(systable) + [newsys]
    newprimetable = list(primetable) + [newfc]
    boo, newsupport = MonomialSupport(newsys, support, vars)
    if not boo:
        return False, False, newsupport, newsystable, newprimetable, 0, 0, []

    newvaluesmod = NewValuesWitness(newsys, vars, support)

    witness, newwitnessmod, newmodulus = WitnessLift(
        newfc, witnessmod, newvaluesmod, modulus
    )
    if len(witness) == 0:
        # witness coefficients could not be lifted
        return True, False, support, newsystable, newprimetable, newwitnessmod, newmodulus, []
    return True, True, support, newsystable, newprimetable, newwitnessmod, newmodulus, witness


def ElimModSatIntersect(eqs1, pol, eqs2, fc, vars, rag_sep_elem, opts=None):
    if opts is None:
        opts = {}
    # Generate a fresh variable name not in vars
    var = "x0"
    i = 1
    while var in [str(v) for v in vars]:
        var = f"u{i}"
        i += 1

    gb1 = MSolveGroebner(
        [var * pol - 1] + list(eqs1), fc, [var] + list(vars),
        {"elim": 1} | opts
    )
    var_set = set(str(v) for v in vars)
    gb1 = [p for p in gb1 if set(str(v) for v in p.variables()) <= var_set]

    gb2 = MSolveGroebner(
        list(gb1) + [pol] + list(eqs2), fc, list(vars) + [rag_sep_elem],
        opts | {"elim": len(vars), "verb": 0}
    )
    if gb2 == [1]:
        return [1]
    gb2 = [p for p in gb2 if set(str(v) for v in p.variables()) == {str(rag_sep_elem)}]

    # Squarefree factorisation mod fc, then take product of factors
    R = gb2[0].parent()
    Rmod = PolynomialRing(GF(fc), R.variable_names())
    sqf = Rmod(gb2[0]).squarefree_decomposition()
    gb2 = prod(factor for factor, _ in sqf)
    gb2 = R(gb2.lift())
    return [gb2]


def ElimModSatIntersectLM(eqs1, pol, eqs2, fc, vars, rag_sep_elem, opts=None):
    if opts is None:
        opts = {}
    var = "x0"
    i = 1
    while var in [str(v) for v in vars]:
        var = f"u{i}"
        i += 1

    gb1 = MSolveGroebner(
        [var * pol - 1] + list(eqs1), fc, [var] + list(vars),
        {"elim": 1} | opts
    )
    var_set = set(str(v) for v in vars)
    gb1 = [p for p in gb1 if set(str(v) for v in p.variables()) <= var_set]

    gb2 = MSolveGroebner(
        list(gb1) + [pol] + list(eqs2), fc, list(vars) + [rag_sep_elem],
        opts | {"elim": len(vars)}
    )
    if gb2 == [1]:
        return [1]
    gb2 = [p for p in gb2 if set(str(v) for v in p.variables()) == {str(rag_sep_elem)}]

    R = gb2[0].parent()
    Rmod = PolynomialRing(GF(fc), R.variable_names())
    sqf = Rmod(gb2[0]).squarefree_decomposition()
    gb2 = prod(factor for factor, _ in sqf)
    gb2 = R(gb2.lift())
    return [_leading_monomial(gb2, [rag_sep_elem])]


def ModSatIntersect(eqs1, pol, eqs2, fc, vs, opts=None):
    if opts is None:
        opts = {}
    var = "x0"
    i = 1
    while var in [str(v) for v in vs]:
        var = f"u{i}"
        i += 1

    R = pol.parent()
    R_ext = PolynomialRing(
        R, [var]
    ).flattening_morphism().codomain()
    var  = R_ext(var)
    gb1 = MSolveGroebner(
        [var * R_ext(pol) - 1] + [R_ext(f) for f in eqs1], fc, [var] + [R_ext(f) for f in vs],
        {"elim": 1} | opts
    )
    var_set = set(str(v) for v in vs)
    gb1 = [R(p) for p in gb1 if set(str(v) for v in p.variables()) <= var_set]
    gb2 = MSolveGroebner(
        list(gb1) + [pol] + list(eqs2), fc, list(vs), opts
    )
    return gb2


def ModSatIntersectLM(eqs1, pol, eqs2, fc, vs, opts=None):
    if opts is None:
        opts = {}
    var = "x0"
    i = 1
    while var in [str(v) for v in vs]:
        var = f"u{i}"
        i += 1
    R = pol.parent()
    R_ext = PolynomialRing(
        R, [var]
    ).flattening_morphism().codomain()
    var  = R_ext(var)
    gb1 = MSolveGroebner(
        [var * R_ext(pol) - 1] + [R_ext(f) for f in eqs1], fc, [var] + [R_ext(f) for f in vs],
        {"elim": 1} | opts
    )
    var_set = set(str(v) for v in vs)
    gb1 = [R(p) for p in gb1 if set(str(v) for v in p.variables()) <= var_set]

    gb2 = MSolveGroebnerLM(
        list(gb1) + [pol] + list(eqs2), fc, list(vs), opts
    )
    return gb2


# Does the same as SaturateIntersect but instead of returning a
# truncated grevlex GB of the solutions, it returns an elimination ideal
def ElimSaturateIntersect(eqs1, pol, eqs2, vars, rag_sep_elem, opts=None):
    if opts is None:
        opts = {}

    import random
    random.seed()

    str_val = opts.get("nthreads", None)
    if isinstance(str_val, int):
        nthreads = str_val
    else:
        nthreads = 1
    newopts = {"linalg": 42, "nthreads": nthreads}

    rr = randint(2**30, 1303905300)
    fcinit = next_prime(rr)
    lm = ElimModSatIntersectLM(eqs1, pol, eqs2, fcinit, vars, rag_sep_elem, newopts)

    fc = next_prime(2**30)
    while fc == fcinit:
        fc = next_prime(fc)
    gb = ElimModSatIntersect(eqs1, pol, eqs2, fc, vars, rag_sep_elem, newopts)
    lmgb = [_leading_monomial(p, list(vars) + [rag_sep_elem]) for p in gb]

    while lmgb != lm:
        fc = next_prime(fc)
        while fc == fcinit:
            fc = next_prime(fc)
        gb = ElimModSatIntersect(eqs1, pol, eqs2, fc, vars, rag_sep_elem, newopts)
        lmgb = [_leading_monomial(p, list(vars) + [rag_sep_elem]) for p in gb]

    fc1 = fc
    sys = list(eqs1) + list(eqs2)
    # N = MinimalGenerators(gb, sys, vars, fc)
    N = 1
    print(f"[deg={max(p.degree() for p in gb[:N])}]", end="", flush=True)

    modulus = fc
    boo, support = MonomialSupport(gb[:N], [[] for _ in range(N)],
                                   list(vars) + [rag_sep_elem])
    boo = True
    primetable = [fc]
    systable = [gb[:N]]
    lctables = [[[] for _ in range(len(support[j]))] for j in range(N)]
    lctables = TableCoeffs(systable[0], list(vars) + [rag_sep_elem], support, lctables)
    if len(lctables) == 0:
        lctables = [[[] for _ in range(len(support[j]))] for j in range(N)]

    witnessmod = NewValuesWitness(gb[:N], list(vars) + [rag_sep_elem], support)
    nprimes = 1
    print(f"{{{nprimes}}}", end="", flush=True)

    oldlifted = []
    oldwitness = []
    islifted = 0   # largest index of polys which are lifted
    lifted = []
    prevlifted = []

    while boo:
        fc = next_prime(fc)
        while fc == fcinit:
            fc = next_prime(fc)

        gb = ElimModSatIntersect(eqs1, pol, eqs2, fc, vars, rag_sep_elem, newopts)
        lmgb = [_leading_monomial(p, list(vars) + [rag_sep_elem]) for p in gb]
        sys = gb[:N]

        if lm == lmgb:
            (boo1, boo2, newsupport, systable, primetable,
             witnessmod, modulus, witness) = GeneratorsLift(
                sys, fc, list(vars) + [rag_sep_elem], modulus, witnessmod,
                support, systable, primetable
            )
            nprimes += 1
            print(f"{{{nprimes}}}", end="", flush=True)
            if not boo1:
                print("[!]", end="", flush=True)
                support = newsupport

            lctables = TableCoeffs(sys, list(vars) + [rag_sep_elem], support, lctables)
            if len(lctables) == 0:
                lctables = [[[] for _ in range(len(support[j]))] for j in range(N)]

            if boo1 and boo2:
                if oldwitness != witness:
                    boo2 = False
                    oldwitness = witness
                else:
                    oldwitness = witness

            if boo1 and boo2:
                print("*", end="", flush=True)
                newlifted = LiftPolynomials(lctables, support, primetable, modulus, islifted)
                islifted = 0
                for i in range(min(len(newlifted), len(prevlifted))):
                    if newlifted[i] == prevlifted[i] and newlifted[i] not in lifted:
                        lifted.append(newlifted[i])
                        print(f"[{len(lifted)}]", end="", flush=True)
                        islifted += 1
                    else:
                        prevlifted = newlifted
                        break
                if len(lifted) == N:
                    return [p.numerator() for p in lifted]
        else:
            print("Bad prime")

    return lifted


def SaturateIntersectMSolve(eqs1, pol, eqs2, vs, opts=None):
    if opts is None:
        opts = {}

    import random
    random.seed()

    str_val = opts.get("nthreads", None)
    if isinstance(str_val, int):
        nthreads = str_val
    else:
        nthreads = 1
    newopts = {"linalg": 42, "nthreads": nthreads}

    rr = randint(2**30, 1303905300)
    fcinit = next_prime(rr)
    lm = ModSatIntersectLM(eqs1, pol, eqs2, fcinit, vs, newopts)
    fc = next_prime(2**30)
    while fc == fcinit:
        fc = next_prime(fc)
    gb = ModSatIntersect(eqs1, pol, eqs2, fc, vs, newopts)
    lmgb = [_leading_monomial(p, vs) for p in gb]

    while lmgb != lm:
        fc = next_prime(fc)
        while fc == fcinit:
            fc = next_prime(fc)
        gb = ModSatIntersect(eqs1, pol, eqs2, fc, vs, newopts)
        lmgb = [_leading_monomial(p, vs) for p in gb]

    fc1 = fc
    sys = list(eqs1) + list(eqs2)
    N = MinimalGenerators(gb, sys, vs, fc)
    #print(f"[ngens={N},mdeg={max(p.degree() for p in gb[:N])}]", end="", flush=True)

    modulus = fc
    boo, support = MonomialSupport(gb[:N], [[] for _ in range(N)], vs)
    boo = True
    primetable = [fc]
    systable = [gb[:N]]
    lctables = [[[] for _ in range(len(support[j]))] for j in range(N)]
    lctables = TableCoeffs(systable[0], vs, support, lctables)
    if len(lctables) == 0:
        lctables = [[[] for _ in range(len(support[j]))] for j in range(N)]

    witnessmod = NewValuesWitness(gb[:N], vs, support)
    nprimes = 1
    #print(f"{{{nprimes}}}", end="", flush=True)

    oldlifted = []
    oldwitness = []
    islifted = 0   # largest index of polys which are lifted
    lifted = []
    prevlifted = []

    while boo:
        if(nprimes>10):
            return
        fc = next_prime(fc)
        while fc == fcinit:
            fc = next_prime(fc)

        gb = ModSatIntersect(eqs1, pol, eqs2, fc, vs, newopts)
        lmgb = [_leading_monomial(p, vs) for p in gb]
        sys = gb[:N]

        if lm == lmgb:
            (boo1, boo2, newsupport, systable, primetable,
             witnessmod, modulus, witness) = GeneratorsLift(
                sys, fc, vs, modulus, witnessmod, support, systable, primetable
            )
            nprimes += 1
            #print(f"{{{nprimes}}}", end="", flush=True)
            if not boo1:
                #print("[!]", end="", flush=True)
                support = newsupport

            lctables = TableCoeffs(sys, vs, support, lctables)
            if len(lctables) == 0:
                lctables = [[[] for _ in range(len(support[j]))] for j in range(N)]

            if boo1 and boo2:
                if oldwitness != witness:
                    boo2 = False
                    oldwitness = witness
                else:
                    oldwitness = witness

            if boo1 and boo2:
                #print("*", end="", flush=True)
                newlifted = LiftPolynomials(lctables, support, primetable, modulus, islifted)
                islifted = 0
                for i in range(min(len(newlifted), len(prevlifted))):
                    if newlifted[i] == prevlifted[i] and newlifted[i] not in lifted:
                        lifted.append(newlifted[i])
                        #print(f"[{len(lifted)}]", end="", flush=True)
                        islifted += 1
                    else:
                        prevlifted = newlifted
                        break
                prevlifted = newlifted
                if len(lifted) == N:
                    return [p.numerator() for p in lifted]
        else:
            print("Bad prime")

    return lifted

# --------------------------------------------------
# ACTUALLY WORKING VERSION -- DO NOT USE ABOVE
# --------------------------------------------------


def SaturateIntersect(eqs1, pol, eqs2, vs, opts=None):
    R = vs[0].parent()
    sat = Ideal(eqs1).saturation(pol)[0]
    print(sat)
    return Ideal(sat).intersection(Ideal(eqs2)).gens()