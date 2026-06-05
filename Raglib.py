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

from helpers import *
from msolve import *

# ---------------------------------------------------------------------------
# ComputeMaximalMinors
# ---------------------------------------------------------------------------

def ComputeMaximalMinors(M):
    """
    Compute the maximal minors of matrix M (nr x nc, nr <= nc).
    Returns a sorted list of square-free minors ordered by degree.
    """
    nr = LinearAlgebra_RowDimension(M)
    nc = LinearAlgebra_ColumnDimension(M)
    if nr > nc:
        raise NotImplementedError("Not implemented yet")

    minors = list(M.minors(nr))

    def squarefree_expand(_pol):
        if degree(_pol) > 0:
            return mul(_p for _p in [_l[0] for _l in sqrfree(_pol)[1]])
        return _pol

    minors = [squarefree_expand(_pol) for _pol in minors]
    minors.sort(key=lambda a: degree(a))
    return minors


# ---------------------------------------------------------------------------
# IsRegular
# ---------------------------------------------------------------------------

def IsRegular(Fs, vs, opts=None):
    """
    Test whether the system F is regular.
    Returns (bool, ld) where ld is the list of maximal minors of the Jacobian.
    """
    if opts is None:
        opts = {}
    rr = random.randint(1, 65520)

    def rr_call():
        return random.randint(1, 65520)

    hyp = sum(rr_call() * _var for _var in vs) + rr_call()
    J = Matrix(
        [
            [f.derivative(v) for v in vs]
            for f in Fs
        ]
    )
    #J = Matrix(linalg_jacobian(Fs, vs))
    ld = ComputeMaximalMinors(J)
    ld = [x for x in ld if x != 0]
    gb = MSolveGroebnerLM([*Fs, *ld, hyp], 0, vs, opts)
    if gb == [1]:
        return True, ld
    else:
        return False, ld


# ---------------------------------------------------------------------------
# HaveFiniteIntersections
# ---------------------------------------------------------------------------

def HaveFiniteIntersections(eqs, cstr, vars, opts=None):
    """
    Test whether each constraint in cstr, together with eqs and a random
    hyperplane, gives a zero-dimensional system.
    """
    if opts is None:
        opts = {}

    def rr():
        return random.randint(1, 2**30)

    for i in range(len(cstr)):
        hyp = sum(rr() * vars[j] for j in range(len(vars))) + rr()
        gb = MSolveGroebnerLM([*eqs, cstr[i], hyp], 0, vars, opts)
        if gb != [1]:
            return False
    return True


# ---------------------------------------------------------------------------
# GoodFiberValue_svars
# ---------------------------------------------------------------------------

def GoodFiberValue_svars(svars, Inequalities, Inequations):
    """
    Find an integer substitution for svars such that no inequality or
    inequation vanishes.
    # To be improved; one should also return values that make positive the
    # inequalities
    """
    boo = True
    j = 0
    while boo:
        spt = {svars[i]: j for i in range(len(svars))}
        sineq = subs(spt, Inequations)
        spos = [
            _pol
            for _pol in [subs(spt, _pol) for _pol in Inequalities]
            if degree(_pol) <= 0
        ]
        if 0 not in sineq and 0 not in spos:
            return spt
        j += 1


# ---------------------------------------------------------------------------
# GoodFiberValue
# ---------------------------------------------------------------------------

def GoodFiberValue(var, hyp, Inequalities, Inequations):
    """
    Find a non-negative integer i such that hyp=i gives a fiber where
    all inequalities and inequations are non-zero.
    Returns (ls, hypsol).
    """
    i = 0
    boo = True
    while boo:
        hypsol = solve(hyp - i, var)
        ls = {var: hypsol}
        pos = subs(ls, Inequalities)
        ineq = subs(ls, Inequations)
        if 0 not in pos and 0 not in ineq:
            return ls, hypsol
        i += 1


# ---------------------------------------------------------------------------
# DegreeTruncate
# ---------------------------------------------------------------------------

def DegreeTruncate(gb, fam, lm, vars, fc, mdeg, opts=None):
    """
    Find the minimal number of elements of gb (up to a degree threshold)
    such that the Groebner basis has the same leading monomials as the
    full computation.
    """
    if opts is None:
        opts = {}

    ldeg = sorted(set(degree(p) for p in gb))
    # Minimum degree in gb
    mgb = min(degree(p) for p in gb)
    tord = tdeg(*vars)
    thresh = min(
        mdeg + (mdeg - max(ldeg)) / 4,
        mgb + (mdeg - mgb) / 4
    )

    for i in range(len(ldeg)):
        deg = ldeg[i]
        if deg >= thresh:
            tgb = [p for p in gb if degree(p) <= deg]
            gb2 = MSolveGroebner([*tgb, *fam], fc, vars, opts)
            oldN = len([p for p in tgb if degree(p) == deg])
            lm2 = [Groebner_LeadingMonomial(p, tord) for p in gb2]
            if lm2 == lm:
                boo = True
                tgb1 = [p for p in tgb if degree(p) < deg]
                tgb2 = [p for p in tgb if degree(p) == deg]
                N = len(tgb2)
                while boo:
                    tgb = [*tgb1, *tgb2[:N // 2]]
                    gb2 = MSolveGroebner([*tgb, *fam], fc, vars, opts)
                    lm2 = [Groebner_LeadingMonomial(p, tord) for p in gb2]
                    if lm2 == lm and N > 1:
                        oldN = N
                        N = min(N, N // 2)
                    else:
                        N = oldN
                        boo = False
                return N + len(tgb1)
    return len(gb)


# ---------------------------------------------------------------------------
# Increment
# ---------------------------------------------------------------------------

def Increment(ol, ind):
    """Increment elements of list ol at positions given by ind (1-based)."""
    l = list(ol)
    for i in range(len(ind)):
        l[ind[i] - 1] += 1
    return l


# ---------------------------------------------------------------------------
# NextForms
# ---------------------------------------------------------------------------

def NextForms(ll):
    """
    Given a list of integer tuples ll, produce the next layer of
    coefficient vectors by incrementing subsets of coordinates.
    """
    n = len(ll[0])
    if n == 1:
        return [*ll, *[[l[0] + 1] for l in ll]]

    newll = []
    for i in range(1, n):
        lp = list(combinations(range(1, n), i))
        for j in range(len(lp)):
            newll.extend([Increment(l, list(lp[j])) for l in ll])
    return newll


# ---------------------------------------------------------------------------
# TestGenericLineDegreeSingular
# ---------------------------------------------------------------------------

def TestGenericLineDegreeSingular(F, vars, hyp, singminors, gendeg, opts=None):
    """
    Compute the degree of the generic line restricted to each singular
    component defined by singminors.
    Returns (ldeg, minors).
    # gendeg is empty when it has not been pre-determined (and hence needs to
    # be computed)
    """
    if opts is None:
        opts = {}

    isbounded = opts.get("isbounded", 0)

    rag_sat_var, rag_hilb_var = SR.var("rag_sat_var, rag_hilb_var")

    if len(F) < len(vars):
        J = Matrix(linalg_jacobian([*F, hyp], vars))
        minors = ComputeMaximalMinors(J)
    else:
        minors = []

    ldeg = []
    toadd = []
    lF = [F[i] - F[0] for i in range(1, len(F))]
    tord = tdeg(*vars)

    for i in range(len(singminors)):
        rr = random.randint(2**30, 1303905301)
        fc = nextprime(rr)
        pol = singminors[i]
        lhyp = [randpoly(vars, degree=2, dense=True) for _ in range(2)]

        if len(gendeg) == 0 or (len(gendeg) > 0 and gendeg[i] > 0):
            mdeg = max(degree(p) for p in [*lF, *minors])
            gb = MSolveGroebner(
                [*lhyp, rag_sat_var * pol - 1, *lF, *minors, *toadd],
                fc,
                [rag_sat_var, *vars],
                {**opts, "elim": 1}
            )
            if gb != [1]:
                deg = -2
            else:
                gb = MSolveGroebner(
                    [rag_sat_var * pol - 1, *lF, *minors, *toadd],
                    fc,
                    [rag_sat_var, *vars],
                    {**opts, "elim": 1}
                )
                gb = [p for p in gb if rag_sat_var not in indets(p)]
                gb2 = MSolveGroebner([*gb, F[0]], fc, vars, opts)
                gb2 = [Groebner_LeadingMonomial(p, tord) for p in gb2]
                hs = Groebner_HilbertSeries(gb2, vars, rag_hilb_var)
                if degree(denom(hs)) == 0:
                    deg = subs({rag_hilb_var: 1}, hs)
                else:
                    deg = -2
        else:
            if gendeg[i] > 0:
                deg = gendeg[i] - 1
            else:
                deg = 0

        ldeg.append(deg)
        toadd.append(pol)

    return ldeg, minors


# ---------------------------------------------------------------------------
# TestGenericLineDegreeRegular
# ---------------------------------------------------------------------------

def TestGenericLineDegreeRegular(F, vars, hyp, opts=None):
    """
    Compute the degree of the generic line for a regular system F.
    Returns (deg, minors).
    """
    if opts is None:
        opts = {}

    rag_hilb_var = SR.var("rag_hilb_var")
    R = extend_ring(vars[0].parent(), rag_hilb_var)
    F, vars, hyp, rag_hilb_var = convert_to_ring(
        R, F, vars, hyp, rag_hilb_var
    )

    if len(F) < len(vars):
        J = Matrix(linalg_jacobian([*F, hyp], vars))
        minors = ComputeMaximalMinors(J)
    else:
        minors = []

    gb = MSolveGroebnerLM([*F, *minors], 0, vars, opts)
    hs = Groebner_HilbertSeries(gb, vars, rag_hilb_var)

    if hs.denominator().degree() == 0:
        deg = subs({rag_hilb_var: 1}, hs)
    else:
        deg = -2
    return deg, minors


# ---------------------------------------------------------------------------
# HasFiniteCriticalLocus
# ---------------------------------------------------------------------------

def HasFiniteCriticalLocus(eqs, Fs, minors, singminors, vs, opts=None):
    """
    Test whether each singular component gives a finite critical locus.
    """
    if opts is None:
        opts = {}

    rag_sat_var = SR.var("rag_sat_var")
    R = PolynomialRing(vs[0].parent(), [rag_sat_var]).flattening_morphism().codomain()
    rag_sat_var = R(rag_sat_var)
    eqs = [R(f) for f in eqs]
    Fs = [R(f) for f in Fs]
    vs = [R(v) for v in vs]
    minors = [R(f) for f in minors]
    singminors = [R(f) for f in singminors]

    def rr():
        return random.randint(1, nextprime(Integer(2**30)))

    hyp = sum(rr() * v for v in vs) + rr()
    for i in range(len(singminors)):
        pol = singminors[i]
        gb = MSolveGroebnerLM(
            [rag_sat_var * pol - 1, *eqs, *Fs, hyp, *minors, *singminors[:i]],
            0,
            [rag_sat_var, *vs],
            {**opts, "linalg": 42}
        )
        if gb != [1]:
            return False
    return True


# ---------------------------------------------------------------------------
# FindGenericLineRegular
# ---------------------------------------------------------------------------

def FindGenericLineRegular(eqs, F, lc, singminors, vars, opts=None):
    """
    Find a generic linear form (hyperplane) for a regular system such that
    the projected system has the generic degree and a finite critical locus.
    Returns (hyp, minors, gendeg).
    """
    if opts is None:
        opts = {}

    isbounded = opts.get("isbounded", 0)
    verb = opts.get("verb", 0)

    dF = [F[i] + lc[i] for i in range(len(F))]

    # Generic degree
    def rr():
        return random.randint(1, nextprime(Integer(2**30)))

    hyp = sum(rr() * v for v in vars)
    gendeg, minors = TestGenericLineDegreeRegular([*eqs, *dF], vars, hyp, opts)
    if gendeg == -2:
        print(eqs, F, lc, singminors, vars)
        raise RuntimeError("Generic line should provide finitely many critical points")

    for i in range(len(vars)):
        if verb >= 1:
            print("+", end="")
        if vars[i] not in F:
            hyp = vars[i]
            deg, minors = TestGenericLineDegreeRegular([*eqs, *dF], vars, hyp, opts)
            if deg == gendeg or (deg >= 0 and isbounded > 0):
                if HasFiniteCriticalLocus(eqs, F, minors, singminors, vars, opts):
                    return hyp, minors, gendeg

    B = 1
    n = len(vars)
    ll = [[1] * n]
    while True:
        for i in range(len(ll)):
            if verb >= 1:
                print("+", end="")
            hyp = sum(ll[i][j] * vars[j] for j in range(n))
            deg, minors = TestGenericLineDegreeRegular([*eqs, *dF], vars, hyp, opts)
            if deg == gendeg or (deg >= 0 and isbounded > 0):
                if HasFiniteCriticalLocus(eqs, F, minors, singminors, vars, opts):
                    return hyp, minors, gendeg
        newll = NextForms(ll)
        ll = [x for x in newll if x not in ll]


# ---------------------------------------------------------------------------
# FindGenericLineSingular
# ---------------------------------------------------------------------------

def FindGenericLineSingular(F, vars, singminors, opts=None):
    """
    Find a generic linear form for a singular system.
    Returns (hyp, minors, gendeg).
    """
    if opts is None:
        opts = {}

    isbounded = opts.get("isbounded", 0)

    def rr():
        return random.randint(1, 65520)

    hyp = sum(rr() * _var for _var in vars)

    if isbounded == 0:
        gendeg, minors = TestGenericLineDegreeSingular(F, vars, hyp, singminors, [], opts)
        if -2 in gendeg:
            raise RuntimeError("Generic line should give finitely many critical points")
    else:
        gendeg = [1] * len(singminors)

    # This is necessary because when F contains algebraically dependent polynomials
    # the next loop will return the first variable (see example
    #   F = [x1,x2,x1-x2],vars=[x1,x2,x3])
    for i in range(len(vars)):
        print("+", end="")
        if vars[i] not in F:
            hyp = vars[i]
            deg, minors = TestGenericLineDegreeSingular(F, vars, hyp, singminors, gendeg, opts)
            all_same_sign = len(set(map(sign, deg))) == 1
            if deg == gendeg or (all_same_sign and isbounded > 0):
                return hyp, minors, gendeg

    B = 1
    n = len(vars)
    ll = [[1] * n]
    while True:
        for i in range(len(ll)):
            print("+", end="")
            hyp = sum(ll[i][j] * vars[j] for j in range(n))
            deg, minors = TestGenericLineDegreeSingular(F, vars, hyp, singminors, gendeg, opts)
            all_same_sign = len(set(map(sign, deg))) == 1
            if deg == gendeg or (all_same_sign and isbounded > 0):
                return hyp, minors, gendeg
        newll = NextForms(ll)
        ll = [x for x in newll if x not in ll]


# ---------------------------------------------------------------------------
# CoeffDeform_eps
# ---------------------------------------------------------------------------

def CoeffDeform_eps(eqs, F, vars, eps, cstr, opts=None):
    """
    Find small integer deformation coefficients lc such that
    the system [eqs, F[i] + lc[i]*eps] has a Groebner basis with
    the same leading monomials as a random deformation and finite
    intersections with cstr.
    """
    if opts is None:
        opts = {}

    def rr():
        return random.randint(1, 2**30)
    
    sat_var = SR.var("sat_var")

    allvars = [sat_var, *vars, eps]
    R = extend_ring(vars[0].parent(), [sat_var])
    eqs, F, vars, eps, cstr, sat_var, allvars = convert_to_ring(
        R, eqs, F, vars, eps, cstr, sat_var, allvars
    )

    tmp = [sat_var * eps - 1, *eqs, *[F[i] + rr() * eps for i in range(len(F))]]
    gb0 = MSolveGroebnerLM(
        [sat_var * eps - 1, *eqs, *[F[i] + rr() * eps for i in range(len(F))]],
        0, allvars,
        {**opts, "linalg": 42}
    )
    lc = [[Integer(1)] * len(F)]
    while True:
        for k in range(len(lc)):
            gb = MSolveGroebnerLM(
                [sat_var * eps - 1, *eqs, *[F[i] + lc[k][i] * eps for i in range(len(F))]],
                0, allvars,
                {**opts, "linalg": 42}
            )
            if gb == gb0:
                if HaveFiniteIntersections(
                    [sat_var * eps - 1, *eqs, *[F[i] + lc[k][i] * eps for i in range(len(F))]],
                    cstr, allvars, opts
                ):
                    return lc[k]
        newlc = NextForms(lc)
        lc = [x for x in newlc if x not in lc]


# ---------------------------------------------------------------------------
# CoeffDeform
# ---------------------------------------------------------------------------

def CoeffDeform(eqs, F, singminors, vars, opts=None):
    """
    Find integer deformation coefficients lc such that the system
    [eqs, F[i]+lc[i]] has the same Hilbert series degree and the
    singular components vanish.
    """
    if opts is None:
        opts = {}
    
    rag_hilb_var = SR.var("rag_hilb_var")

    def rr():
        return random.randint(1, 2**30)

    lhyp = [sum(rr() * vars[i] for i in range(len(vars)))
            for _ in range(len(vars) - len(F) - len(eqs))]
    gb = MSolveGroebnerLM(
        [*eqs, *[F[i] + rr() for i in range(len(F))], *lhyp],
        0, vars,
        {**opts, "linalg": 42}
    )
    hs = Groebner_HilbertSeries(gb, vars, rag_hilb_var)
    if degree(denom(hs)) == 0:
        deg = subs({rag_hilb_var: 1}, hs)
    else:
        print(eqs, F, singminors, vars, opts)
        raise RuntimeError("Bug in CoeffDeform")

    lc = [[1] * len(F)]
    hyp = sum(rr() * vars[i] for i in range(len(vars))) + rr()
    while True:
        for k in range(len(lc)):
            gb = MSolveGroebnerLM(
                [*eqs, *[F[i] + lc[k][i] for i in range(len(F))], *lhyp],
                0, vars,
                {**opts, "linalg": 42}
            )
            gbsing = MSolveGroebnerLM(
                [*eqs, *singminors, *[F[i] + lc[k][i] for i in range(len(F))], hyp],
                0, vars,
                {**opts, "linalg": 42}
            )
            hs = Groebner_HilbertSeries(gb, vars, rag_hilb_var)
            if (degree(denom(hs)) == 0
                    and subs({rag_hilb_var: 1}, hs) == deg
                    and gbsing == [1]):
                return lc[k]
        newlc = NextForms(lc)
        lc = [x for x in newlc if x not in lc]


# ---------------------------------------------------------------------------
# FindGenericLine
# ---------------------------------------------------------------------------

def FindGenericLine(eqs, F, vars, opts=None):
    """
    Top-level routine: determine regularity, compute a coefficient
    deformation, and find a generic linear form.
    Returns (hyp, minors_or_vminors, gendeg, is_regular, lc).
    """
    if opts is None:
        opts = {}

    verb = opts.get("verb", 0)

    boo, singminors = IsRegular([*eqs, *F], vars, opts)
    lc = CoeffDeform(eqs, F, singminors, vars, opts)
    hyp, minors, gendeg = FindGenericLineRegular(eqs, F, lc, singminors, vars, opts)

    if boo is True:
        if verb >= 1:
            print("[R]", end="")
        return hyp, minors, gendeg, True, lc
    else:
        if verb >= 1:
            print("[S]", end="")
        return hyp, [minors, singminors], gendeg, False, lc


# ---------------------------------------------------------------------------
# SmallMidRational
# ---------------------------------------------------------------------------

def SmallMidRational(rr1, rr2):
    """
    Find a rational number strictly between rr1 and rr2 with a small
    continued-fraction representation.
    """
    if rr1 >= rr2:
        print(rr1, rr2)
        raise RuntimeError("Bug detected")
    mid = (rr1 + rr2) / 2
    c = continued_fraction(mid).convergents()
    for ci in c[:20]:
        if rr1 < ci < rr2:
            return ci
    return mid


# ---------------------------------------------------------------------------
# ConstructFibers
# ---------------------------------------------------------------------------

def ConstructFibers(ll, hyp, cstr):
    """
    Given a sorted list of intervals ll, construct sample fiber values
    (one per gap, plus one below and one above) avoiding zeros in cstr.
    """
    vvar = hyp.variables()[0]
    R = vvar.parent()

    # Smallest fiber
    val = math.floor(ll[0][0]) - 1
    ls = {vvar: solve_line(hyp - val, vvar)}
    if 0 in subs(ls, cstr):
        return ConstructFibers([[val, val], *ll], hyp, cstr)

    # Largest fiber
    val = math.ceil(ll[-1][1]) + 1
    ls = {vvar: solve_line(hyp - val, vvar)}
    if 0 in subs(ls, cstr):
        return ConstructFibers([*ll, [val, val]], hyp, cstr)

    # Remaining ones
    for i in range(len(ll) - 1):
        val = (ll[i][1] + ll[i + 1][0]) / 2
        ls = {vvar: solve_line(hyp - val, vvar)}
        if 0 in subs(ls, cstr):
            new_ll = sorted([[val, val], *ll], key=lambda a: a[1])
            return ConstructFibers(new_ll, hyp, cstr)

    # Construct the fibers
    res = [math.floor(ll[0][0] - 1)]
    for i in range(len(ll) - 1):
        mid = SmallMidRational(ll[i][1], ll[i + 1][0])
        res.append(mid)
    res.append(math.ceil(ll[-1][1] + 1))
    return res


# ---------------------------------------------------------------------------
# HasOverLapCoupleOfIntervals
# ---------------------------------------------------------------------------

def HasOverLapCoupleOfIntervals(l1, l2):
    """Return True if intervals l1 and l2 overlap."""
    return l1[1] >= l2[0]


# ---------------------------------------------------------------------------
# HasOverLap
# ---------------------------------------------------------------------------

def HasOverLap(_list):
    """
    Return True if any two consecutive intervals in the sorted list overlap.
    Assumes that _list has been sorted.
    """
    if len(_list) <= 1:
        return False
    for i in range(len(_list) - 1):
        boo = HasOverLapCoupleOfIntervals(_list[i], _list[i + 1])
        if boo is True:
            return boo
    return False


# ---------------------------------------------------------------------------
# ComputeBoundsRegular
# ---------------------------------------------------------------------------

def ComputeBoundsRegular(Equations, Fam, Positive, NotNull, vars,
                         hyp, minors, gendeg, opts=None):
    """
    Compute sample fiber values for a regular system by finding real roots
    of the projected system and constructing fiber values in each gap.
    Returns (hyp, fiber_values).
    """
    rag_sep_elem = SR.var("rag_sep_elem")
    if opts is None:
        opts = {}

    R = extend_ring(vars[0].parent(), [rag_sep_elem])
    Equations, Fam, Positive, NotNull, vars, hyp, minors, rag_sep_elem = convert_to_ring(
        R, Equations, Fam, Positive, NotNull, vars, hyp, minors, rag_sep_elem
    )


    verb = opts.get("verb", 0)
    rd = random.randint(1, 65521)

    if gendeg == 0:
        sols = [0, []]
    else:
        if len(Equations) + len(Fam) < len(vars):
            sols = MSolveRealRoots(
                [*Equations, *Fam, *minors, hyp - rag_sep_elem],
                [*vars, rag_sep_elem],
                [*Positive, *NotNull],
                opts
            )
        else:
            sols = MSolveRealRoots(
                [*Equations, *Fam, *minors, hyp - rag_sep_elem],
                [*vars, rag_sep_elem],
                [],
                opts
            )
        if sols[0] > 0:
            print(Equations, Fam, Positive, NotNull, vars, hyp, minors, gendeg, opts)
            raise RuntimeError("Degenerate case in ComputeBounds: to be implemented (bounds)")
        sols = [0, AdmissibleSolutions(sols, len(Positive))]

    if len(sols[1]) > 0:
        rr = sorted(set(subs(s, rag_sep_elem) for s in sols[1]),
                    key=lambda a: a[1] if hasattr(a, '__getitem__') else a)
        if not HasOverLap(rr):
            rr = ConstructFibers(rr, hyp, [*Positive, *NotNull])
        else:
            if verb >= 1:
                print("[Overlap Regular]", end="")
            gb = MSolveGroebner(
                [*Equations, *Fam, *minors, hyp - rag_sep_elem],
                0,
                [*vars, rag_sep_elem],
                {**opts, "elim": len(vars)}
            )
            sols = MSolveRealRoots(gb, [rag_sep_elem], [])
            rr = sorted(set(subs(s, rag_sep_elem) for s in sols[1]),
                        key=lambda a: a[1] if hasattr(a, '__getitem__') else a)
            rr = ConstructFibers(rr, hyp, [*Positive, *NotNull])
        return hyp, rr
    else:
        i = 0

        vvar = R(SR(hyp).variables()[0])
        ls = {vvar: R(solve(SR(hyp - i), SR(vvar))[0].rhs())}

        def rr_rand():
            return random.randint(1, 65521)

        lF = [*Equations, *[_p + rr_rand() for _p in Fam], hyp - i]
        lhyp = [randpoly(vars, degree=1, dense=True)
                for _ in range(len(vars) - len(lF) + 1)]
        gb = MSolveGroebnerLM([*lhyp, *lF], 0, vars, opts)
        while gb != [1] or 0 in subs(ls, [*Positive, *NotNull, *Fam]):
            i += 1
            lF = [*Equations, *[_p + rr_rand() for _p in Fam], hyp - i]
            lhyp = [randpoly(vars, degree=1, dense=True)
                    for _ in range(len(vars) - len(lF) + 1)]
            gb = MSolveGroebnerLM([*lhyp, *lF], 0, vars, opts)
            ls = {vvar: R(solve(SR(hyp - i), SR(vvar))[0].rhs())}
        return hyp, [i]


# ---------------------------------------------------------------------------
# ManageOverLapComputeBoundsSingular
# ---------------------------------------------------------------------------

def ManageOverLapComputeBoundsSingular(Equations, Fam, singminors, minors,
                                       gendeg, lgb, hyp, Positive, NotNull,
                                       vars, opts=None):
    """
    Handle the overlap case for singular bounds by computing a univariate
    polynomial whose real roots separate the intervals.
    Returns fiber values.
    """
    if opts is None:
        opts = {}

    rag_sep_elem, rag_sat_var = SR.var("rag_sep_elem, rag_sat_var")

    toadd = []
    upol = 1
    for i in range(len(singminors)):
        pol = singminors[i]
        if gendeg[i] != 0:
            gb1 = MSolveGroebner(
                [*lgb[i], *Equations, *minors, *Fam, *toadd, hyp - rag_sep_elem],
                0, [*vars, rag_sep_elem],
                {**opts, "elim": len(vars)}
            )
            upol = upol * gb1[0]
            gb2 = MSolveGroebner(
                [rag_sat_var * pol - 1, *Equations, *minors, *Fam, *toadd, hyp - rag_sep_elem],
                0, [rag_sat_var, *vars, rag_sep_elem],
                {**opts, "elim": len(vars) + 1}
            )
            upol = upol * gb2[0]
        toadd.append(pol)

    squpol = [_l[0] for _l in sqrfree(upol)[1]]
    if len(squpol) == 0:
        print(Equations, Fam, singminors, minors, gendeg, lgb, hyp, Positive, NotNull, vars, opts)
        raise RuntimeError("Bug detected (ManageOverLapComputeBoundsSingular)")

    upol = mul(*squpol)
    sols = MSolveRealRoots([upol], [rag_sep_elem], [])
    sols = sols[1]
    rr = sorted(set(subs(s, rag_sep_elem) for s in sols),
                key=lambda a: a[1] if hasattr(a, '__getitem__') else a)
    rr = ConstructFibers(rr, hyp, [*Positive, *NotNull])
    return rr


# ---------------------------------------------------------------------------
# ElimComputeBoundsSingular
# ---------------------------------------------------------------------------

def ElimComputeBoundsSingular(Equations, Fam, Positive, NotNull, vars,
                               hyp, vminors, gendeg, opts=None):
    """
    Compute sample fiber values for a singular system via elimination.
    Returns (hyp, fiber_values).
    """
    if opts is None:
        opts = {}

    verb = opts.get("verb", 0)

    rag_sep_elem, rag_sat_var = SR.var("rag_sep_elem, rag_sat_var")

    def rd():
        return random.randint(1, 65521)

    singminors = vminors[1]
    minors = vminors[0]
    sols = []
    toadd = []
    lF = [Fam[i] - Fam[0] for i in range(1, len(Fam))]
    lgb = []

    for i in range(len(singminors)):
        pol = singminors[i]
        if gendeg[i] == 0:
            nsols = [0, []]
            lgb.append([1])
        else:
            OldDigits = get_precision()
            new_digits = max(
                10,
                max(degree(p) for p in map(expand, [*Fam, *Equations]))
                + int(max(ilog2(abs(c))
                          for p in map(expand, [*Fam, *Equations])
                          for c in coeffs(p)) / 2)
            )
            set_precision(new_digits)

            gb = ElimSaturateIntersect(
                [*Equations, *minors, *lF, *toadd],
                pol,
                [pol, Fam[0], *singminors, hyp - rag_sep_elem],
                vars, rag_sep_elem, opts
            )
            lgb.append(gb)
            nsols = [0, []]
            nsols2 = MSolveRealRoots(
                [rag_sat_var * pol - 1, *Equations, *minors, *Fam, *toadd, hyp - rag_sep_elem],
                [rag_sat_var, *vars, rag_sep_elem],
                [], opts
            )
            if len(nsols2) < 2:
                print(Equations, Fam, Positive, NotNull, vars, hyp, vminors, gendeg, opts)
                raise RuntimeError("nsols2 should have cardinality 2 (2)")
            else:
                nsols2 = [0, AdmissibleSolutions(nsols2, len(Positive))]
                nsols2 = [0, [
                    [c for c in s if indets(c) <= indets([*vars, rag_sep_elem])]
                    for s in nsols2[1]
                ]]
            set_precision(OldDigits)
            nsols = [0, [*nsols[1], *nsols2[1]]]

        if len(nsols) < 2:
            print(Equations, Fam, Positive, NotNull, vars, hyp, vminors, gendeg, opts)
            raise RuntimeError("nsols should have cardinality 2")
        toadd.append(pol)
        sols.extend(nsols[1])

    if len(sols) > 0:
        rr = sorted(set(subs(s, rag_sep_elem) for s in sols),
                    key=lambda a: a[1] if hasattr(a, '__getitem__') else a)
        if not HasOverLap(rr):
            rr = ConstructFibers(rr, hyp, [*Positive, *NotNull])
        else:
            if verb >= 1:
                print("Overlap Singular", end="")
            rr = ManageOverLapComputeBoundsSingular(
                Equations, Fam, singminors, minors,
                gendeg, lgb, hyp, Positive, NotNull, vars, opts
            )
        return hyp, rr
    else:
        j = 0
        vvar = list(indets(hyp))[0]
        if max(gendeg) == 0:
            ls = {vvar: solve(hyp - j, vvar)}
            while 0 in subs(ls, [*Fam, *Positive, *NotNull]):
                j += 1
                ls = {vvar: solve(hyp - j, vvar)}
            return hyp, [j]

        def rr():
            return random.randint(1, 65521)

        lF = [*Equations, *[_p + rr() for _p in Fam], hyp - j]
        lhyp = [randpoly(vars, degree=1, dense=True)
                for k in range(len(vars) - len(lF) + 1)]
        gb = MSolveGroebnerLM([*lhyp, *lF], 0, vars, opts)
        ls = {vvar: solve(hyp - j, vvar)}
        while gb != [1] or 0 in subs(ls, [*Positive, *NotNull, *Fam]):
            j += 1
            lF = [*Equations, *[_p + rr() for _p in Fam], hyp - j]
            lhyp = [randpoly(vars, degree=1, dense=True)
                    for k in range(len(vars) - len(lF) + 1)]
            gb = MSolveGroebnerLM([*lhyp, *lF], 0, vars, opts)
            ls = {vvar: solve(hyp - j, vvar)}
        return hyp, [j]


# ---------------------------------------------------------------------------
# ComputeBoundsSingular
# ---------------------------------------------------------------------------

def ComputeBoundsSingular(Equations, Fam, Positive, NotNull, vars,
                           hyp, vminors, gendeg, lc, opts=None):
    """
    Compute sample fiber values for a singular system.
    Returns (hyp, fiber_values).
    """
    if opts is None:
        opts = {}

    verb = opts.get("verb", 0)

    rag_sep_elem, rag_sat_var = SR.var("rag_sep_elem, rag_sat_var")

    def rd():
        return random.randint(1, 65521)

    minors = vminors[0]
    singminors = [x for x in vminors[1] if x != 0]
    singminors = [x for x in singminors if x not in minors]

    sols = []
    toadd = []
    lF = [lc[0] * Fam[i] - lc[i] * Fam[0] for i in range(1, len(Fam))]
    lgb = []

    for i in range(len(singminors)):
        pol = singminors[i]
        if gendeg[i] == 0:
            nsols = [0, []]
            lgb.append([1])
        else:
            OldDigits = get_precision()
            new_digits = max(
                10,
                max(degree(p) for p in map(expand, [*Fam, *Equations]))
                + int(max(ilog2(abs(c))
                          for p in map(expand, [*Fam, *Equations])
                          for c in coeffs(p)) / 2)
            )
            set_precision(new_digits)

            gb = SaturateIntersect(
                [*Equations, *minors, *lF, *toadd],
                pol,
                [pol, Fam[0]],
                vars, opts
            )
            lgb.append(gb)

            if len(Equations) + len(Fam) < len(vars):
                nsols = MSolveRealRoots(
                    [*gb, *Equations, *minors, *Fam, *toadd, hyp - rag_sep_elem],
                    [*vars, rag_sep_elem],
                    [*Positive, *NotNull], opts
                )
            else:
                nsols = MSolveRealRoots(
                    [*gb, *Equations, *minors, *Fam, *toadd, hyp - rag_sep_elem],
                    [*vars, rag_sep_elem],
                    [], opts
                )
            if len(nsols) < 2:
                print(Equations, Fam, Positive, NotNull, vars, hyp, vminors, gendeg, lc, opts)
                raise RuntimeError("nsols should have cardinality 2 (1)")

            nsols = [0, AdmissibleSolutions(nsols, len(Positive))]

            if len(Equations) + len(Fam) < len(vars):
                nsols2 = MSolveRealRoots(
                    [rag_sat_var * pol - 1, *Equations, *minors, *Fam, *toadd, hyp - rag_sep_elem],
                    [rag_sat_var, *vars, rag_sep_elem],
                    [*Positive, *NotNull], opts
                )
            else:
                nsols2 = MSolveRealRoots(
                    [rag_sat_var * pol - 1, *Equations, *minors, *Fam, *toadd, hyp - rag_sep_elem],
                    [rag_sat_var, *vars, rag_sep_elem],
                    [], opts
                )
            if len(nsols2) < 2:
                print(Equations, Fam, Positive, NotNull, vars, hyp, vminors, gendeg, lc, opts)
                raise RuntimeError("nsols2 should have cardinality 2 (2)")
            else:
                nsols2 = [0, AdmissibleSolutions(nsols2, len(Positive))]
                nsols2 = [0, [
                    [c for c in s if indets(c) <= indets([*vars, rag_sep_elem])]
                    for s in nsols2[1]
                ]]
            set_precision(OldDigits)
            nsols = [0, [*nsols[1], *nsols2[1]]]

        if len(nsols) < 2:
            print(Equations, Fam, Positive, NotNull, vars, hyp, vminors, gendeg, lc, opts)
            raise RuntimeError("nsols should have cardinality 2")
        toadd.append(pol)
        sols.extend(nsols[1])

    if len(sols) > 0:
        rr = sorted(set(subs(s, rag_sep_elem) for s in sols),
                    key=lambda a: a[1] if hasattr(a, '__getitem__') else a)
        if not HasOverLap(rr):
            rr = ConstructFibers(rr, hyp, [*Positive, *NotNull])
        else:
            if verb >= 1:
                print("Overlap Singular", end="")
            rr = ManageOverLapComputeBoundsSingular(
                Equations, Fam, singminors, minors,
                gendeg, lgb, hyp, Positive, NotNull, vars, opts
            )
        return hyp, rr
    else:
        j = 0
        vvar = list(indets(hyp))[0]
        if max(gendeg) == 0:
            ls = {vvar: solve(hyp - j, vvar)}
            while 0 in subs(ls, [*Fam, *Positive, *NotNull]):
                j += 1
                ls = {vvar: solve(hyp - j, vvar)}
            return hyp, [j]

        def rr():
            return random.randint(1, 65521)

        lF = [*Equations, *[_p + rr() for _p in Fam], hyp - j]
        lhyp = [randpoly(vars, degree=1, dense=True)
                for k in range(len(vars) - len(lF) + 1)]
        gb = MSolveGroebnerLM([*lhyp, *lF], 0, vars, opts)
        ls = {vvar: solve(hyp - j, vvar)}
        while gb != [1] or 0 in subs(ls, [*Positive, *NotNull, *Fam]):
            j += 1
            lF = [*Equations, *[_p + rr() for _p in Fam], hyp - j]
            lhyp = [randpoly(vars, degree=1, dense=True)
                    for k in range(len(vars) - len(lF) + 1)]
            gb = MSolveGroebnerLM([*lhyp, *lF], 0, vars, opts)
            ls = {vvar: solve(hyp - j, vvar)}
        return hyp, [j]


# ---------------------------------------------------------------------------
# ComputeBounds
# ---------------------------------------------------------------------------

def ComputeBounds(Equations, Fam, Positive, NotNull, vars, opts=None):
    """
    Top-level routine: find a generic line, then compute bounds via the
    regular or singular path.
    Returns (hyp, fiber_values).
    """
    if opts is None:
        opts = {}

    hyp, minors, gendeg, boo, lc = FindGenericLine(Equations, Fam, vars, opts)
    if boo:
        return ComputeBoundsRegular(Equations, Fam, Positive, NotNull, vars,
                                    hyp, minors, gendeg, opts)
    else:
        return ComputeBoundsSingular(Equations, Fam, Positive, NotNull, vars,
                                     hyp, minors, gendeg, lc, opts)


# ---------------------------------------------------------------------------
# ModularLimits
# ---------------------------------------------------------------------------

def ModularLimits(Equations, pol, vars, eps, opts=None):
    """
    Compute the degree truncation needed for limits of critical points
    near the vanishing locus of pol.
    Returns (bool, trunc_deg).
    """
    if opts is None:
        opts = {}

    rag_sat_var = SR.var("rag_sat_var")

    rrfc = random.randint(2**30, 1303905301)
    fc = nextprime(rrfc)
    mdeg = max(degree(p) for p in [pol, *Equations])
    gb = MSolveGroebner(
        [rag_sat_var * pol - 1, *Equations],
        fc,
        [rag_sat_var, *vars],
        {**opts, "elim": 1}
    )
    gb = [p for p in gb if rag_sat_var not in indets(p)]
    gb2 = MSolveGroebner([eps, *gb], fc, vars, opts)
    if gb2 == [1]:
        return False, 0
    else:
        tord = tdeg(*vars)
        gb2 = [Groebner_LeadingMonomial(p, tord) for p in gb2]
        trunc_deg = DegreeTruncate(gb, [eps], gb2, vars, fc, mdeg, opts)
        return True, trunc_deg


# ---------------------------------------------------------------------------
# LimitsDeformedCriticalPoints
# ---------------------------------------------------------------------------

def LimitsDeformedCriticalPoints(Equations, Fam, Inequalities,
                                   Inequations, pol1, pol2, vs, opts=None):
    """
    Compute limiting critical points as epsilon -> 0 for a deformed family.
    Returns a list of solutions.
    """
    if opts is None:
        opts = {}

    rag_eps_var, rag_sat_var = SR.var("rag_eps_var, rag_sat_var")

    def rr():
        return random.randint(1, 65520)

    lf = [sum(rr() * v for v in vs) for _ in range(len(Fam))]
    dFam = [Fam[i] + rag_eps_var * lf[i] for i in range(len(Fam))]
    hyp1 = sum(rr() * v for v in vs)
    hyp2 = sum(rr() * v for v in vs)

    J1 = Matrix(linalg_jacobian([*Equations, *dFam, hyp1], vs))
    J2 = Matrix(linalg_jacobian([*Equations, *dFam, hyp2], vs))
    JS = Matrix(linalg_jacobian([*Equations, *dFam], vs))

    sminors = ComputeMaximalMinors(JS)
    if len(Equations) + len(Fam) < len(vs):
        minors = list(set(ComputeMaximalMinors(J1)) | set(ComputeMaximalMinors(J2)))
    else:
        minors = []

    sminors = sorted(
        [m for m in sminors if m not in minors],
        key=lambda a: degree(a)
    )

    toadd = []
    sols = []
    lminors = SplitSystem(minors)

    for i in range(len(sminors)):
        pol = sminors[i]
        for j in range(len(lminors)):
            boo, N = ModularLimits(
                [*Equations, *dFam, *toadd, *lminors[j]],
                pol * rag_eps_var,
                [rag_eps_var, *vs],
                rag_eps_var, opts
            )
            if boo is True:
                gb = MSolveGroebner(
                    [*Equations, *dFam,
                     rag_sat_var * pol * rag_eps_var - 1,
                     *lminors[j], *toadd],
                    0,
                    [rag_sat_var, rag_eps_var, *vs],
                    {**opts, "trunc": N, "elim": 1}
                )
                nsols = MSolveRealRoots(
                    [rag_sat_var * Fam[0] - 1, *gb, pol, *toadd],
                    [rag_sat_var, *vs, rag_eps_var],
                    [pol1, *Inequalities, pol2, *Inequations],
                    opts
                )
                if len(nsols) < 2:
                    print(nsols)
                    raise RuntimeError("nsols should have cardinality 2")
                nsols = AdmissibleSolutions(nsols, len(Inequalities) + 1)
                nsols = [
                    {v:k for v, k in sol.items() if v in vs}
                    for sol in nsols
                ]
            else:
                nsols = []
            toadd.append(pol)
            sols.extend(nsols)

    return sols


# ---------------------------------------------------------------------------
# CriticalPointsSingular
# ---------------------------------------------------------------------------

def CriticalPointsSingular(Equations, Fam, Inequalities, Inequations, pol,
                            pol2, vs, minors, opts=None):
    """
    Compute critical points on singular components, first trying to obtain
    a finite set by removing constraint vanishing loci, then falling back
    to LimitsDeformedCriticalPoints.
    Returns a list of solutions.
    """
    if opts is None:
        opts = {}

    rag_sat_var1, rag_sat_var2, rag_hilb_var = SR.var("rag_sat_var1, rag_sat_var2, rag_hilb_var")

    # One first tries to remove from the set of singular points vanishing loci of
    # constraints to see if we get finitely many points (turned out to be useful in
    # many applications).
    cstr = [*Inequalities, *Inequations]
    toremove = []
    for i in range(len(cstr)):
        gb = MSolveGroebnerLM(
            [*Equations,
             *[Fam[j] - Fam[0] for j in range(len(Fam))],
             *minors,
             rag_sat_var1 * Fam[0] - 1,
             rag_sat_var2 * cstr[0] - 1],
            0,
            [rag_sat_var2, rag_sat_var1, *vs],
            {"elim": 2, **opts}
        )
        hs = Groebner_HilbertSeries(gb, vs, rag_hilb_var)
        if degree(denom(hs)) == 0:
            toremove = [cstr[i]]
            break

    if len(toremove) > 0:
        sols = MSolveRealRoots(
            [*Equations,
             *[Fam[i] - Fam[0] for i in range(1, len(Fam))],
             *minors,
             rag_sat_var1 * Fam[0] - 1,
             rag_sat_var2 * toremove[0] - 1],
            [rag_sat_var2, rag_sat_var1, *vs],
            [pol, *Inequalities, pol2, *Inequations],
            opts
        )
        sols = AdmissibleSolutions(sols, len(Inequalities) + 1)
        sols = [{v:k for v, k in _p.items() if v in vs} for _p in sols]
        return sols

    return LimitsDeformedCriticalPoints(
        Equations, Fam, Inequalities, Inequations, pol, pol2, vs, opts
    )


# ---------------------------------------------------------------------------
# SplitSystem_cstr
# ---------------------------------------------------------------------------

def SplitSystem_cstr(F, cstr):
    """
    Split F into irreducible factor systems, removing factors divisible by
    any element of cstr, and filtering by Groebner normal form w.r.t.
    degree-1 polynomials in each system.
    """
    if len(F) == 0:
        return F
    if len(F) == 1:
        pol = F[0]
        if degree(pol) <= 0:
            return [[pol]]
        else:
            lf = [_l[0] for _l in factors(pol)[1]]
            lf = [
                [_l] for _l in lf
                if not any(divide(_p, _l) for _p in cstr)
            ]
            return lf

    pol = F[0]
    if degree(pol) > 0:
        lf = [primpart(_l[0]) for _l in factors(pol)[1]]
        lf = [_l for _l in lf if not any(divide(_p, _l) for _p in cstr)]
    else:
        lf = [pol]

    lsys = list(set(
        tuple(sorted(s)) for s in SplitSystem(F[1:])
    ))
    lsys = [list(s) for s in lsys]
    lsys = [[*_l, _p] for _l in lsys for _p in lf]

    vars_all = list(indets([*F, *cstr]))
    newlsys = []
    for i in range(len(lsys)):
        sys = lsys[i]
        degone = [_p for _p in sys if degree(_p) == 1]
        if 0 not in [Groebner_NormalForm(_p, degone, tdeg(*vars_all)) for _p in cstr]:
            newlsys.append(sys)
    return newlsys


# ---------------------------------------------------------------------------
# SplitSystem
# ---------------------------------------------------------------------------

def SplitSystem(F):
    """
    Split F into all combinations of irreducible factors (one factor per
    polynomial in F), without repetitions within a combination.
    """
    if len(F) == 0:
        return F
    if len(F) == 1:
        pol = F[0]
        if degree(pol) <= 0:
            return [[pol]]
        else:
            return [[_l[0]] for _l in factors(pol)[1]]

    pol = F[0]
    if degree(pol) > 0:
        lf = [primpart(_l[0]) for _l in factors(pol)[1]]
    else:
        lf = [pol]

    lsys = list(set(tuple(sorted(s)) for s in SplitSystem(F[1:])))
    lsys = [list(s) for s in lsys]
    result = []
    for _l in lsys:
        for _p in lf:
            if _p not in _l:
                result.append([*_l, _p])
    return result


# ---------------------------------------------------------------------------
# CriticalPoints
# ---------------------------------------------------------------------------

def CriticalPoints(Equations, Fam, Inequalities, Inequations, vs, opts=None):
    """
    Compute all critical points satisfying the constraints.
    Returns a list of solutions.
    """
    if opts is None:
        opts = {}

    rag_sat_var = SR.var("rag_sat_var")

    J = Matrix(linalg_jacobian([*Equations, *Fam], vs))
    minors = ComputeMaximalMinors(J)

    positive = [p for p in Inequalities if p not in Fam]
    pos_pol = sorted(
        [p for p in Inequalities if p in Fam],
        key=lambda a: degree(a)
    )
    pol = pos_pol[0] if len(pos_pol) > 0 else 1

    nz_pol = [p for p in Inequations if p in Fam or -p in Fam]
    pol2 = nz_pol[0] if len(nz_pol) > 0 else 1

    np = len(positive)
    cstr = [
        *[p for p in Inequalities if p not in Fam],
        *[p for p in Inequations if p not in Fam and -p not in Fam]
    ]

    sysminors = SplitSystem_cstr(minors, cstr)
    sysminors = [sys for sys in sysminors if 0 not in map(degree, sys)]
    sysminors = [[p for p in sys if p != 0] for sys in sysminors]
    sysminors = [sys for sys in sysminors
                 if len(set(sys) & set(Inequations)) == 0]
    sysminors = [sys for sys in sysminors
                 if len(set(sys) & set(Inequalities)) == 0]

    sols = []
    for j in range(len(sysminors)):
        newsols = MSolveRealRoots(
            [*Equations,
             *[Fam[i] - Fam[0] for i in range(1, len(Fam))],
             rag_sat_var * Fam[0] - 1,
             *sysminors[j]],
            [rag_sat_var, *vs],
            [pol, *positive,
             pol2, *[p for p in Inequations if p not in Fam and -p not in Fam]],
            opts
        )
        if newsols == [1]:
            newsols = CriticalPointsSingular(
                Equations, Fam, positive,
                nz_pol, pol, pol2, vs, sysminors[j], opts
            )
        else:
            newsols = AdmissibleSolutions(newsols, np + 1)

        newsols = [
            {v:k for v,k in pt.items() if v in vs}
            for pt in newsols
        ]
        sols.extend(newsols)

    return sols


# ---------------------------------------------------------------------------
# ExactSolSelection
# Warning: ce n'est correct que sur les solutions exactes
# ---------------------------------------------------------------------------

def ExactSolSelection(sols, Inequalities, Inequations, vars):
    """
    Filter solutions that violate any inequality or inequation
    (using midpoints of the interval representation).
    """
    badsols = []
    for i in range(len(sols)):
        osol = sols[i]
        sol = {v: (k[0] + k[1]) / 2 for v, k in osol.items()}
        if (any(subs(sol, p) <= 0 for p in Inequalities) or
                any(subs(sol, p) == 0 for p in Inequations)):
            badsols.append(osol)
    return [s for s in sols if s not in badsols]


# ---------------------------------------------------------------------------
# GenerateCriticalPointsFamilies
# ---------------------------------------------------------------------------

def GenerateCriticalPointsFamilies(FamPositive, FamNotNull):
    """
    Generate all families obtained by appending +pol or -pol for each
    element of FamNotNull.
    """
    lFam = [FamPositive]
    for j in range(len(FamNotNull)):
        pol = FamNotNull[j]
        lFam1 = [_l + [pol] for _l in lFam]
        lFam2 = [_l + [-pol] for _l in lFam if len(_l) > 0]
        lFam = [*lFam1, *lFam2]
    return lFam


# ---------------------------------------------------------------------------
# FamCriticalPoints
# ---------------------------------------------------------------------------

def FamCriticalPoints(Equations, FamPositive, FamNotNull,
                      Inequalities, Inequations, vars, opts=None):
    """
    Compute critical points over all generated families.
    Returns a list of solutions.
    """
    if opts is None:
        opts = {}

    sols = []
    Families = GenerateCriticalPointsFamilies(FamPositive, FamNotNull)
    for i in range(len(Families)):
        Fam = sorted(Families[i], key=lambda a: degree(a))
        sols.extend(
            CriticalPoints(Equations, Fam, Inequalities, Inequations, vars, opts)
        )
    return sols


# ---------------------------------------------------------------------------
# UnivariateSolveFamily
# ---------------------------------------------------------------------------

def UnivariateSolveFamily(Equations, Fam, Inequalities, Inequations, vs):
    """
    Solve a univariate polynomial system satisfying constraints.
    Returns a list of solutions.
    """
    R = vs[0].parent()
    g = R(0)
    for i in range(len(Equations)):
        g = gcd(g, Equations[i])
    if degree(g) == 0:
        return []

    if len(Equations) > 0:
        for i in range(len(Inequalities)):
            q = gcd(g, Inequalities[i])
            if degree(q) > 0:
                g = numer(normal(g / q))
        if degree(g) == 0:
            return []

        for i in range(len(Inequations)):
            q = gcd(g, Inequations[i])
            if degree(q) > 0:
                g = numer(normal(g / q))
            g = gcd(g, Inequations[i])

    if degree(g) == 0:
        return []

    # Now, we search for real roots of g which satisfy the constraints.
    if degree(g) > 0:
        sols = MSolveRealRoots(
            [g],
            list(indets(g)),
            [*Inequalities, *Inequations]
        )
        sols = AdmissibleSolutions(sols, len(Inequalities))
        return sols

    # Case where g = 0
    upol = mul(Fam)
    for i in range(len(Equations)):
        upol = gcd(upol, Equations[i])
    sq = sqrfree(upol)[1]
    if len(sq) > 0:
        upol = mul(*[p[0] for p in sq])
    else:
        upol = 1

    newpol = 1
    for i in range(len(Equations)):
        newpol = expand(newpol * Equations[i])
        if degree(newpol) > 1:
            sq = sqrfree(newpol)[1]
            newpol = mul([p[0] for p in sq])
    for i in range(len(Fam)):
        newpol = expand(newpol * Fam[i])
        if degree(newpol) > 1:
            sq = sqrfree(newpol)[1]
            newpol = mul([p[0] for p in sq])
    for i in range(len(Inequalities)):
        newpol = expand(newpol * Inequalities[i])
        if degree(newpol) > 1:
            sq = sqrfree(newpol)[1]
            newpol = mul([p[0] for p in sq])
    for i in range(len(Inequations)):
        newpol = expand(newpol * Inequations[i])
        if degree(newpol) > 1:
            sq = sqrfree(newpol)[1]
            newpol = mul([p[0] for p in sq])
    newpol = expand(newpol)

    uroots = MSolveRealRoots([newpol], vs, [])[1]
    uroots = [sol[vs[0]] for sol in uroots]
    uroots = sorted(uroots, key=lambda a: a[1]) # TODO should technically be a[1] < b[0]
    if HasOverLap(uroots):
        print(newpol)
        raise RuntimeError("Bug in msolve")

    sols = []
    for i in range(len(uroots) - 1):
        mid = SmallMidRational(uroots[i][1], uroots[i + 1][0])
        if -1 not in [sign(subs({vs[0]: mid}, p)) for p in Inequalities]:
            sols.append({vs[0]: [mid, mid]})

    if len(uroots) > 0:
        mid = math.floor(uroots[0][0] - 1)
        if -1 not in [sign(subs({vs[0]: mid}, p)) for p in Inequalities]:
            sols = [{vs[0]: [mid, mid]}, *sols]
        mid = math.ceil(uroots[-1][1] + 1)
        if -1 not in [sign(subs({vs[0]: mid}, p)) for p in Inequalities]:
            sols.append({vs[0]: [mid, mid]})
    else:
        if -1 not in [sign(subs({vs[0]: 0}, p)) for p in Inequalities]:
            sols = [{vs[0]: [0, 0]}]

    return sols


# ---------------------------------------------------------------------------
# AdmissibleSolutions
# ---------------------------------------------------------------------------

def AdmissibleSolutions(tsols, np):
    """
    Filter tsols[1] keeping only solutions where the first np sign
    conditions are positive and the remaining are non-zero.
    """
    if len(tsols) == 2:
        return tsols[1]
    elif len(tsols) == 0:
        return []

    sols = []
    if len(tsols[1]) > 0 and len(tsols) > 2:
        for j in range(len(tsols[1])):
            # sgn = [op(x) for x in tsols[2][j][:np]]
            # sgn := map(op, tsols[3][j][1..np]);
            sgn = [
                x
                for arr in tsols[2][j][:np]
                for x in arr
            ]
            if (-1 not in [sign(x) for x in sgn]
                    and 0 not in sgn
                    and 0 not in [x for xs in tsols[2][j][np:] for x in xs]):
                sols.append(tsols[1][j])
    return sols


# ---------------------------------------------------------------------------
# GenerateDeformedFamilies_eps
# Generates families to solve
# ---------------------------------------------------------------------------

def GenerateDeformedFamilies_eps(eqs, FamPositive, FamNotNull,
                                  vars, eps, cstr, opts=None):
    """
    Generate all deformed families by appending +pol or -pol per element
    of FamNotNull, then computing CoeffDeform_eps for each.
    """
    if opts is None:
        opts = {}

    deform = [FamPositive]
    for i in range(len(FamNotNull)):
        pol = FamNotNull[i]
        deform1 = [_l + [pol] for _l in deform]
        deform2 = [_l + [-pol] for _l in deform if len(_l) > 0]
        deform = [*deform1, *deform2]

    lsys = []
    for i in range(len(deform)):
        lc = CoeffDeform_eps(eqs, deform[i], vars, eps, cstr, opts)
        sys = [*eqs, *[deform[i][j] - lc[j] * eps for j in range(len(deform[i]))]]
        lsys.append(sys)

    return lsys


# ---------------------------------------------------------------------------
# OLDGenerateDeformedFamilies_eps
# Generates families to solve
# ---------------------------------------------------------------------------

def OLDGenerateDeformedFamilies_eps(FamPositive, FamNotNull, lcpos, lcineq, eps):
    """
    (Older version) Generate deformed families using pre-computed
    coefficient lists lcpos and lcineq.
    """
    deform = [[FamPositive[i] - lcpos[i] * eps for i in range(len(lcpos))]]
    for i in range(len(FamNotNull)):
        pol = FamNotNull[i]
        deform1 = [_l + [pol - lcineq[i] * eps] for _l in deform]
        deform2 = [_l + [pol + lcineq[i] * eps] for _l in deform if len(_l) > 0]
        deform = [*deform1, *deform2]
    return deform


# ---------------------------------------------------------------------------
# ConstrainedValues
# ---------------------------------------------------------------------------

def ConstrainedValues(Equations, FamPositive, FamNotNull, vs,
                       Inequalities, Inequations, opts=None):
    """
    Compute constrained values by solving deformed zero-dimensional systems.
    Returns a list of solutions.
    """
    if opts is None:
        opts = {}

    eps, _l = SR.var("eps, _l")

    deform = GenerateDeformedFamilies_eps(Equations, FamPositive, FamNotNull, vs, eps)
    sols = []
    npos = len(Inequalities)

    def rr():
        return random.randint(1, 65520)

    for i in range(len(deform)):
        if indets(deform[i]) != indets(vs):
            mvars = list(set(indets(vs)) - set(indets(deform[i])))
        else:
            mvars = []

        tsols = MSolveRealRoots(
            [*mvars, *Equations, *deform[i]],
            [eps, *vs],
            [*Inequalities, *Inequations],
            opts
        )
        if tsols[0] > 0 or tsols[0] == -1:
            tsols = MSolveRealRoots(
                [_l * eps - 1, *mvars, *Equations, *deform[i]],
                [_l, eps, *vs],
                [*Inequalities, *Inequations],
                opts
            )
            if tsols[0] > 0:
                print(Equations, FamPositive, FamNotNull, vs,
                      Inequalities, Inequations, opts)
                raise RuntimeError("Deformed systems are not 0-dim")

        if len(Inequalities) > 0:
            tsols = AdmissibleSolutions(tsols, npos)
        else:
            if len(Inequations) > 0:
                newtsols = []
                for j in range(len(tsols[1])):
                    for k in range(len(tsols[2][j])):
                        if tsols[2][j][k][0] == 0 and tsols[2][j][k][1] == 0:
                            break
                        if tsols[2][j][k][0] == 0 and tsols[2][j][k][1] != 0:
                            print([[_l * eps - 1, *mvars, *Equations,
                                    *[coeff(_p, eps, 0) - rr() * eps for _p in deform[i]]],
                                   [_l, eps, *vs], [*Inequalities, *Inequations], opts])
                            raise RuntimeError("Bug in MSolveRealRoots to be investigated")
                        if tsols[2][j][k][0] != 0 and tsols[2][j][k][1] == 0:
                            print([[_l * eps - 1, *mvars, *Equations,
                                    *[coeff(_p, eps, 0) - rr() * eps for _p in deform[i]]],
                                   [_l, eps, *vs], [*Inequalities, *Inequations], opts])
                            raise RuntimeError("Bug in MSolveRealRoots to be investigated")
                tsols = newtsols
            else:
                tsols = tsols[1]

        sols.extend(tsols)

    if len(FamPositive) > 0:
        sols = [_p for _p in sols if subs(_p, eps)[0] > 0]
    sols = [{v:k for v, k in _p.items() if v in vs} for _p in sols]
    return sols


# ---------------------------------------------------------------------------
# UnboundedComponents
# ---------------------------------------------------------------------------

def UnboundedComponents(Equations, FamPositive, FamNotNull, Inequalities,
                         Inequations, vars, opts=None):
    """
    Compute sample points in unbounded semi-algebraic components by
    projecting onto a fiber, then recursing on each fiber value.
    Returns a list of solutions.
    """
    if opts is None:
        opts = {}

    verb = opts.get("verb", 0)
    isempty = opts.get("isempty", 0)

    sols = []
    Fam = sorted([*FamPositive, *FamNotNull], key=lambda a: degree(a))
    Positive = [p for p in Inequalities if p not in FamPositive]
    NotNull = [p for p in Inequations if p not in FamNotNull]

    if verb >= 1:
        print("[b:", end="")
    hyp, bounds = ComputeBounds(Equations, Fam, Positive, NotNull, vars, opts)
    if verb >= 1:
        print(f" -> {hyp}, {bounds}", end="")
    if verb >= 1:
        print("]", end="")

    R = vars[0].parent()

    for i in range(len(bounds)):
        v = R(SR(hyp).variables()[0])
        hypsol = R(solve(SR(hyp - bounds[i]), SR(v))[0].rhs())
        ls = {v: hypsol}
        NewEquations = [expand(subs(ls, p)) for p in Equations]
        NewFamPositive = [expand(subs(ls, p)) for p in FamPositive]
        NewFamNotNull = [expand(subs(ls, p)) for p in FamNotNull]
        NewInequations = [expand(subs(ls, p)) for p in Inequations]
        NewInequalities = [expand(subs(ls, p)) for p in Inequalities]
        newvars = [x for x in vars if x != v]
        tsols = SolveFamily(NewEquations, NewFamPositive, NewFamNotNull,
                            NewInequalities, NewInequations, newvars, opts)

        if degree(hypsol) <= 0:
            sols.extend([{v: [hypsol, hypsol], **s} for s in tsols])
        else:
            sols.extend([
                {v: [subs({v: k[0] for v, k in s.items()}, hypsol),
                     subs({v: k[1] for v, k in s.items()}, hypsol)],
                 **{k: val for k, val in s.items()}}
                for s in tsols
            ])
        if len(sols) > 0 and isempty > 0:
            return sols

    return sols


# ---------------------------------------------------------------------------
# DegenerateDeformedSystem
# ---------------------------------------------------------------------------

def DegenerateDeformedSystem(sys, ld, Inequalities, Inequations, vars, eps, opts):
    """
    Handle a degenerate deformed system by saturating and computing real roots.
    Returns real root solutions.
    """
    gb = SaturateIntersect(sys, ld[0], ld, [*vars, eps], opts)
    sols = MSolveRealRoots(
        [*gb, *sys, ld[0]],
        [*vars, eps],
        [*Inequalities, eps, *Inequations],
        opts
    )
    return sols


# ---------------------------------------------------------------------------
# InfiniteBranches
# ---------------------------------------------------------------------------

def InfiniteBranches(sys, ld, Inequalities, Inequations, vs, eps, opts=None):
    """
    Compute solutions near infinity (as eps -> 0) for a deformed system.
    Returns (sols1, sols2).
    """
    if opts is None:
        opts = {}

    T_, rag_sat_var, rag_sep = SR.var("T_, rag_sat_var, rag_sep")
    R = extend_ring(vs[0].parent(), [T_, rag_sat_var, rag_sep])
    sys, ld, Inequalities, Inequations, vs, eps, T_, rag_sat_var, rag_sep = convert_to_ring(
        R, sys, ld, Inequalities, Inequations, vs, eps, T_, rag_sat_var, rag_sep
    )

    verb = opts.get("verb", 0)
    isbounded = opts.get("isbounded", 0)

    allvars = [*vs, eps]

    def rr():
        return random.randint(2**16, 2**30)

    hyp = sum(rr() * allvars[i] for i in range(len(allvars))) + rr()
    gb = MSolveGroebnerLM(
        [rag_sat_var * eps - 1, *sys, hyp],
        0, [rag_sat_var, *allvars],
        {**opts, "elim": 1}
    )
    hs = Groebner_HilbertSeries(gb, allvars, T_)
    deg = abs(QQ(subs({T_: 1}, numer(hs))))
    dim = degree(denom(hs))

    if dim > 0:
        sys0 = SaturateIntersect(sys, ld[0], [], allvars, opts)
        gb0 = MSolveGroebnerLM(
            [hyp, rag_sat_var * eps - 1, *sys0],
            0, [rag_sat_var, *allvars],
            {**opts, "elim": 1}
        )
    else:
        sys0 = sys
        gb0 = gb

    n = len(allvars)
    ll = [[1] * n]
    boo = True
    while boo:
        for i in range(len(ll)):
            if verb >= 1:
                print("[+]", end="")
            hyp = sum(ll[i][j] * allvars[j] for j in range(n))
            gb = MSolveGroebnerLM(
                [rag_sat_var * eps - 1, hyp + rr(), *sys0],
                0, [rag_sat_var, *allvars],
                {**opts, "elim": 1}
            )
            if gb == gb0:
                boo = False
        newll = NextForms(ll)
        ll = [x for x in newll if x not in ll]

    sols = MSolveRealRoots(
        [rag_sep - hyp, *sys, eps],
        [*allvars, rag_sep],
        [], opts
    )

    if sols[0] > 0:
        gb = SaturateIntersect(sys, ld[0], [eps], allvars, opts)
        sols = MSolveRealRoots(
            [*gb, eps, *sys, rag_sep - hyp],
            [*allvars, rag_sep],
            []
        )

    sols[1] = [
        {R(v): k for v, k in sol.items()}
        for sol in sols[1]
    ]
    spec = [
        abs(x)
        for _p in sols[1]
        for x in subs(_p, rag_sep)
    ]
    if len(spec) > 0:
        smin = math.floor(min(spec)) - 1
        smax = math.ceil(max(spec)) + 1
        sols1 = MSolveRealRoots(
            [hyp - smin, *sys0, rag_sat_var * eps - 1],
            [rag_sat_var, *allvars],
            [*Inequalities, eps, *Inequations],
            opts
        )
        sols2 = MSolveRealRoots(
            [hyp - smax, *sys0, rag_sat_var * eps - 1],
            [rag_sat_var, *allvars],
            [*Inequalities, eps, *Inequations],
            opts
        )
    else:
        sols1 = [-1, []]
        sols2 = MSolveRealRoots(
            [hyp - 1, *sys0, rag_sat_var * eps - 1],
            [rag_sat_var, *allvars],
            [*Inequalities, eps, *Inequations],
            opts
        )

    return sols1, sols2


# ---------------------------------------------------------------------------
# ZeroDimBoundaries
# ---------------------------------------------------------------------------

def ZeroDimBoundaries(Equations, FamPositive, FamNotNull,
                       Inequalities, Inequations, vs, opts=None):
    """
    Compute boundary points where the semi-algebraic set has dimension zero
    by deforming the family and tracking specialisations.
    Returns a list of solutions.
    """
    if opts is None:
        opts = {}

    sols1, sols2 = [], []

    eps, rag_sat_var = SR.var("eps, rag_sat_var")
    R = extend_ring(vs[0].parent(), [eps, rag_sat_var])
    Equations, FamPositive, FamNotNull, Inequalities, Inequations, vs, eps, rag_sat_var = convert_to_ring(
        R, Equations, FamPositive, FamNotNull, Inequalities, Inequations, vs, eps, rag_sat_var
    )

    verb = opts.get("verb", 0)
    isbounded = opts.get("isbounded", 0)

    maxdeg = max(degree(p) for p in [*Equations, *FamPositive, *FamNotNull])
    lsys = GenerateDeformedFamilies_eps(
        Equations, FamPositive, FamNotNull, vs, eps,
        [*Inequalities, *Inequations]
    )

    # Inequations that are NOT part of FamNotNull — those are already
    # deformed inside lsys[i] and must not be appended again.
    extra_ineqs = [p for p in Inequations if p not in FamNotNull
                                          and -p not in FamNotNull]
    Inequations = extra_ineqs

    J = Matrix(linalg_jacobian([*Equations, *FamPositive, *FamNotNull], vs))
    delta = ComputeMaximalMinors(J)
    if delta == [0]:
        return []

    lsols = []
    for i in range(len(lsys)):
        if maxdeg > 1:
            sols = MSolveRealRoots(
                [rag_sat_var * eps - 1, *lsys[i], *delta],
                [rag_sat_var, eps, *vs],
                [*Inequalities, eps, *Inequations],
                opts
            )
            if sols[0] > 0:
                sols = DegenerateDeformedSystem(
                    lsys[i], delta, Inequalities, Inequations, vs, eps, opts
                )
                if sols[0] > 0:
                    print(Equations, FamPositive, FamNotNull,
                          Inequalities, Inequations, vs, opts)
                    raise RuntimeError("Bug in ZeroDimBoundaries (1)")
            if len(FamPositive) > 0:
                sols = AdmissibleSolutions(sols, len(Inequalities) + 1)
            else:
                sols = AdmissibleSolutions(sols, len(Inequalities))
            sols = [{v:k for v,k in _p.items() if v in vs} for _p in sols]
            lsols.extend(sols)

        emin = 2
        for j in range(len(Inequalities)):
            sols = MSolveRealRoots(
                [rag_sat_var * eps - 1, *lsys[i], Inequalities[j]],
                [rag_sat_var, eps, *vs],
                [], opts
            )
            if sols[0] > 0:
                print(Equations, FamPositive, FamNotNull,
                      Inequalities, Inequations, vs, opts)
                raise RuntimeError("Bug in ZeroDimBoundaries")
            if len(sols[1]) > 0:
                emin = min(
                    abs(x)
                    for _p in sols[1]
                    for x in subs(_p, eps) # TODO - subs
                )
                # emin = min(abs(op(subs(_p, eps))) for _p in sols[1])
                #emin:=min(map(abs, map(op, map(_p->subs(_p, eps), sols[2]))));

        # TODO - figure out 
        print("TODO:")
        print("ineq:", Inequations)
        for j in range(len(Inequations)):
            sols = MSolveRealRoots(
                [rag_sat_var * eps - 1, *lsys[i], Inequations[j]],
                [rag_sat_var, eps, *vs],
                [], opts
            )
            if sols[0] > 0:
                print(Equations, FamPositive, FamNotNull,
                      Inequalities, Inequations, vs, opts)
                raise RuntimeError("Bug in ZeroDimBoundaries")
            if len(sols[1]) > 0:
                emin = min(
                    abs(x)
                    for _p in sols[1]
                    for x in subs(_p, eps) # TODO - subs
                )

        # Find rational approximation of emin
        emin = AA(emin).n().nearby_rational(emin/2**10)/2
        def _solve_at_emin(emin_val):
            
            R = eps.parent()
            return MSolveRealRoots(
                subs({eps: emin_val}, lsys[i]),
                vs,
                [*Inequalities, *Inequations],
                opts
            )

        if emin == 2:
            sols = _solve_at_emin(emin)
            while sols[0] > 0:
                if verb >= 1:
                    print("*", end="")
                emin /= 2
                sols = _solve_at_emin(emin)
        else:
            emin /= 2
            sols = _solve_at_emin(emin)
            while sols[0] > 0:
                if verb >= 1:
                    print("*", end="")
                emin /= 2
                sols = _solve_at_emin(emin)

        sols = AdmissibleSolutions(sols, len(Inequalities))
        sols = [{v:k for v,k in _p.items() if v in vs} for _p in sols]
        lsols.extend(sols)

        # Additional specialisations of eps when all constraints are inequations
        if len(FamPositive) == 0:
            if emin == 2:
                sols = _solve_at_emin(-emin)
                while sols[0] > 0:
                    if verb >= 1:
                        print("*", end="")
                    emin /= 2
                    sols = _solve_at_emin(-emin)
            else:
                emin /= 2
                sols = _solve_at_emin(-emin)
                while sols[0] > 0:
                    if verb >= 1:
                        print("*", end="")
                    emin /= 2
                    sols = _solve_at_emin(-emin)

            sols = AdmissibleSolutions(sols, len(Inequalities))
            sols = [{v:k for v,k in _p.items() if v in vs} for _p in sols]
            lsols.extend(sols)

        if maxdeg > 1 and isbounded == 0:
            sols1, sols2 = InfiniteBranches(
                lsys[i], delta, Inequalities, Inequations, vs, eps, opts
            )
        if len(FamPositive) > 0:
            sols = AdmissibleSolutions(sols1, len(Inequalities) + 1)
        else:
            sols = AdmissibleSolutions(sols1, len(Inequalities))
        sols = [{v:k for v,k in _p.items() if v in vs} for _p in sols]
        lsols.extend(sols)

        if len(FamPositive) > 0:
            sols = AdmissibleSolutions(sols2, len(Inequalities) + 1)
        else:
            sols = AdmissibleSolutions(sols2, len(Inequalities))
        sols = [{v:k for v,k in _p.items() if v in vs} for _p in sols]
        lsols.extend(sols)

    return lsols


# ---------------------------------------------------------------------------
# SolveFamily
# ---------------------------------------------------------------------------

def SolveFamily(Equations, FamPositive, FamNotNull, Inequalities,
                Inequations, vars, opts=None):
    """
    Main solver for one family: dispatch to univariate, square system,
    UnboundedComponents, ZeroDimBoundaries, or ConstrainedValues.
    Returns a list of solutions.
    """
    if opts is None:
        opts = {}

    verb = opts.get("verb", 0)
    isempty = opts.get("isempty", 0)
    isbounded = opts.get("isbounded", 0)

    if (0 in [*FamPositive, *Inequalities] or
            0 in [*FamNotNull, *Inequations]):
        return []
    if -1 in [sign(_pol) for _pol in
              [_pol for _pol in [*FamPositive, *Inequalities]
               if degree(_pol) <= 0]]:
        return []

    Fam = sorted([*FamPositive, *FamNotNull], key=lambda a: degree(a))
    sols = []

    if len(vars) == 1:
        return UnivariateSolveFamily(Equations, FamPositive, Inequalities,
                                     Inequations, vars)

    if len(Equations) == len(vars):
        sols = MSolveRealRoots(
            Equations, vars,
            [*FamPositive, *Inequalities, *FamNotNull, *Inequations],
            opts
        )
        if sols[0] > 0:
            raise RuntimeError("Bug detected")
        sols = AdmissibleSolutions(sols, len(FamPositive) + len(Inequalities))
        return sols

    if len(Equations) + len(Fam) < len(vars):
        sols = UnboundedComponents(
            Equations, FamPositive, FamNotNull, Inequalities, Inequations, vars, opts
        )
        if len(sols) > 0 and isempty > 0:
            return sols

    if len(Equations) + len(Fam) == len(vars):
        sols = ZeroDimBoundaries(
            Equations, FamPositive, FamNotNull, Inequalities, Inequations, vars, opts
        )
        if len(sols) > 0 and isempty > 0:
            return sols

    if False and len(Equations) + len(Fam) > len(vars):
        if verb >= 1:
            print("[cn", end="")
        try:
            sols = ConstrainedValues(
                Equations, FamPositive, FamNotNull, vars,
                [p for p in Inequalities if p not in FamPositive],
                [p for p in Inequations if p not in FamNotNull],
                opts
            )
        except Exception:
            print(FamPositive, FamNotNull, vars, opts)
            raise RuntimeError("Problem when calling ConstrainedValues")
        if verb >= 1:
            print("]", end="")
        if len(sols) > 0 and isempty > 0:
            return sols
    return sols


# ---------------------------------------------------------------------------
# SemiAlgebraicSolveIterateOnFamilies
# ---------------------------------------------------------------------------

def SemiAlgebraicSolveIterateOnFamilies(Equations, Families, Inequalities,
                                         Inequations, vars, opts):
    """
    Iterate SolveFamily over all candidate families, collecting solutions.
    Returns a list of solutions.
    """
    verb = opts.get("verb", 0)
    isempty = opts.get("isempty", 0)
    newvars = opts.get("newvars", [])
    card = opts.get("card", -1)

    sols = []
    st = time()

    for i in range(len(Families)):
        Fam = Families[i]
        pos = [Inequalities[idx] for idx in Fam[0]]
        nonzero = [Inequations[idx] for idx in Fam[1]]

        cond1 = (len(newvars) == 0 and len(Fam[0]) + len(Fam[1]) >= card)
        cond2 = (indets([*Equations, *pos, *nonzero]) == set(newvars))
        if cond1 or cond2:
            if verb >= 1:
                print("<", end="")
            newsols = SolveFamily(
                Equations, pos, nonzero,
                Inequalities, Inequations, vars, opts
            )
            if verb >= 1:
                print(f":->[ns={len(newsols)}]:", end="")
            if verb >= 1 and (time() - st >= 10 or i == len(Families) - 1):
                print(f"{{{'%.2f' % (100.0 * (i + 1) / len(Families))}%}}", end="")
                st = time()
            if verb >= 1:
                print(">", end="")

            sols.extend(newsols)
            if isempty > 0 and len(sols) > 0:
                return sols

    if verb >= 1:
        print(f"++{{{len(sols)}}}++", end="")
    return sols


# ---------------------------------------------------------------------------
# PointsPerComponentsAlgebraic
# ---------------------------------------------------------------------------

def PointsPerComponentsAlgebraic(Equations, Inequalities, Inequations, opts=None):
    """
    Compute one sample point per connected component of the algebraic set
    defined by Equations, satisfying Inequalities and Inequations.
    Returns a list of solutions.
    """
    if opts is None:
        opts = {}

    isempty = opts.get("isempty", 0)

    if (0 in Inequations or 0 in Inequalities or
            -1 in [sign(_pol) for _pol in
                   [_pol for _pol in Inequalities if degree(_pol) == 0]]):
        return []

    vs = list(indets(Equations))

    # Check that Equations is regular enough
    boo, singminors = IsRegular(Equations, vs)
    if boo is False:
        raise NotImplementedError("Singular case not implemented yet")

    svars = [v for v in list(set(indets(Inequalities)) | set(indets(Inequations)))
             if v not in vs]
    spt = GoodFiberValue_svars(svars, Inequalities, Inequations)
    spos = subs(spt, Inequalities)
    sineq = subs(spt, Inequations)

    if len(vs) == len(Equations):
        if -1 in [sign(_pol) for _pol in
                  [_pol for _pol in spos if degree(_pol) <= 0]]:
            return []
        sols = MSolveRealRoots(Equations, vs, [*spos, *sineq], opts)
        sols = AdmissibleSolutions(sols, len(spos))
        sols = [[*_sol, *list(spt.items())] for _sol in sols]
        return sols

    # Find a good choice of projection
    hyp, minors, gendeg = FindGenericLineRegular(
        Equations, [], [], singminors, vs, opts
    )

    # Compute the associated critical points and select those satisfying
    # the constraints
    if gendeg == 0:
        return [0, []]

    sols = MSolveRealRoots([*Equations, *minors], vs, [*spos, *sineq], opts)
    sols = AdmissibleSolutions(sols, len(spos))
    sols = [[*_sol, *list(spt.items())] for _sol in sols]

    if len(sols) > 0 and isempty > 0:
        return sols

    # Recursive call on an arbitrary fiber
    var = list(indets(hyp))[0]
    ls, hypsol = GoodFiberValue(var, hyp, Inequalities, Inequations)
    newsols = PointsPerComponentsAlgebraic(
        subs(ls, Equations),
        subs(ls, Inequalities),
        subs(ls, Inequations),
        opts
    )
    if degree(hypsol) <= 0:
        newsols = [{var: [hypsol, hypsol], **s} for s in newsols]
    else:
        newsols = [
            {var: [subs({v: k[0] for v, k in s.items()}, hypsol),
                   subs({v: k[1] for v, k in s.items()}, hypsol)],
             **{k: v for k, v in s.items()}}
            for s in newsols
        ]
    newsols = [[*_sol, *list(spt.items())] for _sol in newsols]

    if len(newsols) > 0 and isempty > 0:
        return newsols

    return [*sols, *newsols]


# ---------------------------------------------------------------------------
# SemiAlgebraicSolve
# ---------------------------------------------------------------------------

def SemiAlgebraicSolve(Equations, Inequalities, Inequations, opts=None):
    """
    Top-level semi-algebraic solver. Computes sample points per component
    of the algebraic set and iterates over all sign conditions for the
    constraints.
    Returns a list of solutions.
    """
    if opts is None:
        opts = {}

    verb = opts.get("verb", 0)
    isempty = opts.get("isempty", 0)

    vs = indets([*Equations, *Inequalities, *Inequations])
    R = PolynomialRing(QQ, vs)
    vs = [R(v) for v in vs]
    Equations = [R(f) for f in Equations]
    Inequalities = [R(f) for f in Inequalities]
    Inequations = [R(f) for f in Inequations]

    sols = []
    lsigns = set()

    if len(Equations) > 0:
        sols = PointsPerComponentsAlgebraic(Equations, Inequalities, Inequations, opts)
        if isempty >= 1 and len(sols) > 0:
            return sols

    oldvars, = convert_to_ring(R, indets(Equations))
    nc = len(Inequalities) + len(Inequations)
    _Studied = [[[], []]]
    newsols = []

    # TODO: improve the choice of Inequalities and introduce criteria to
    # reduce the combinatorial complexity
    for i in range(len(Inequalities)):
        pol = Inequalities[i]
        newvars = list(indets(pol))
        _toStudy = [
            [[*l[0], i], []]
            for l in _Studied
            if len(l[0]) + len(l[1]) <= len(vs)
        ]

        if verb >= 1:
            print(f"\nDealing with (positivity) constraint {i + 1} / {nc} "
                  f"of degree {pol.degree()}")
            print(f"Number of systems to study is {len(_toStudy)}")

        if max(p.degree() for p in [*Equations, pol, *Inequalities[:i]]) == 1:
            newopts = {**opts, "card": i + 1}
        else:
            newopts = opts

        # If pol involves variables which were not appearing previously and
        # which appear in degree one
        # the study of some subfamilies can be skipped
        if len(oldvars) > 0 and "card" not in newopts:
            new_only = [v for v in newvars if v not in oldvars]
            if pol.degree(new_only) == 1:
                newopts = {**opts, "newvars": indets(pol) | indets(oldvars)}
                if newopts == opts:
                    oldnewvars = new_only
            else:
                newopts = opts

        newsols = SemiAlgebraicSolveIterateOnFamilies(
            Equations, _toStudy, Inequalities, Inequations, vs, newopts
        )

        midsols = [
            {v: (k[0] + k[1]) / 2 for v,k in pt.items()}
            for pt in newsols
        ]
        lsigns = lsigns | set(
            tuple(map(sign, [q.subs(pt) for q in Inequations]))
            for pt in midsols
        )
        if verb >= 1 and len(Inequations) > 0:
            print(f"Signs of inequations at computed points: {lsigns}")

        sols.extend(newsols)
        _Studied = [*_Studied, *_toStudy]
        if isempty >= 1 and len(sols) > 0:
            return sols
        oldvars = list(indets(pol) | indets(oldvars))

    npos = len(Inequalities)

    # TODO: improve the choice of Inequations and introduce criteria to
    # reduce the combinatorial complexity
    for i in range(len(Inequations)):
        pol = Inequations[i]
        newvars = list(pol.variables())
        _toStudy = [
            [l[0], [*l[1], i]]
            for l in _Studied
            if len(l[0]) + len(l[1]) <= len(vs)
        ]

        if verb >= 1:
            print(f"\nDealing with (non-zero) constraint "
                  f"{len(Inequalities) + i + 1} / {nc} of degree {pol.degree()}")
            print(f"Number of systems to study is {len(_toStudy)}")

        if max(p.degree() for p in [*Equations, pol, *Inequalities, *Inequations[:i]]) == 1:
            newopts = {**opts, "card": i + npos + 1}
        else:
            newopts = opts

        # If pol involves variables which were not appearing previously and
        # which appear in degree one (or if all polynomials have degree one),
        # the study of some subfamilies can be skipped
        if len(oldvars) > 0 and "card" not in newopts:
            new_only = [v for v in newvars if v not in oldvars]
            if max(sum(term[1].degree(v) for v in new_only) for term in pol) == 1:
                newopts = {**opts, "newvars": indets(pol) | indets(oldvars)}
            else:
                newopts = opts

        newsols = SemiAlgebraicSolveIterateOnFamilies(
            Equations, _toStudy, Inequalities, Inequations, vs, newopts
        )
        midsols = [
            {R(v): (k[0] + k[1]) / 2 for v, k in pt.items()}
            for pt in newsols
        ]

        lsigns = lsigns | set(
            tuple(map(sign, [q.subs(pt) for q in Inequations]))
            for pt in midsols
        )
        if verb >= 1 and len(Inequations) > 0:
            print(f"Signs of inequations at computed points: {lsigns}")

        # Inequations[i] has no sign change, it can be removed
        if len(set(s[i] for s in lsigns)) == 1:
            if verb >= 1:
                print("Removes non-zero constraint", Inequations[i])
            _toStudy = [_l for _l in _toStudy if Inequations[i] not in _l]

        for sol in newsols:
            if sol not in sols:
                sols.append(sol)
        #sols = list(set(tuple(sorted(s.items())) for s in [*sols, *newsols]))
        #sols = [dict(s) for s in sols]

        if isempty >= 1 and len(sols) > 0:
            return sols

        _Studied = [*_Studied, *_toStudy]
        #debug(loop = "", i = i, sols = sols)
        oldvars = list(set(oldvars) | set(R(v) for v in SR(pol).variables()))

    return sols