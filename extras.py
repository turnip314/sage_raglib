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

# Extra functions in RAGlib that don't seem to ever be used in the main routines


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
# SplitSystem_cstr
# ---------------------------------------------------------------------------

def SplitSystem_cstr(F, cstr):
    """
    Split F into irreducible factor systems, removing factors divisible by
    any element of cstr, and filtering by Groebner normal form w.r.t.
    degree-1 polynomials in each system.
    """
    debug(reached="reached SplitSystem_cstr")
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

    vars_all = list(set([x for f in [*F, *cstr] for x in f.variables()]))
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
# ElimComputeBoundsSingular
# ---------------------------------------------------------------------------

def ElimComputeBoundsSingular(Equations, Fam, Positive, NotNull, vars,
                               hyp, vminors, gendeg, opts=None):
    """
    Compute sample fiber values for a singular system via elimination.
    Returns (hyp, fiber_values).
    """
    # TODO - this function is never used?
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
            """
            OldDigits = get_precision()
            new_digits = max(
                10,
                max(degree(p) for p in map(expand, [*Fam, *Equations]))
                + int(max(ilog2(abs(c))
                          for p in map(expand, [*Fam, *Equations])
                          for c in coeffs(p)) / 2)
            )
            set_precision(new_digits)
            """

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
                    [c for c in s if set(c.keys()) <= set([*vars, rag_sep_elem])]
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
        vvar = list(hyp.variables())[0]
        m = max(gendeg) if type(gendeg) == list or type(gendeg) == tuple else gendeg
        if m == 0:
            ls = {vvar: solve_line(hyp - j, vvar)}
            while 0 in subs(ls, [*Fam, *Positive, *NotNull]):
                j += 1
                ls = {vvar: solve_line(hyp - j, vvar)}
            return hyp, [j]

        def rr():
            return random.randint(1, 65521)

        lF = [*Equations, *[_p + rr() for _p in Fam], hyp - j]
        lhyp = [randpoly(vars, degree=1, dense=True)
                for k in range(len(vars) - len(lF) + 1)]
        gb = MSolveGroebnerLM([*lhyp, *lF], 0, vars, opts)
        ls = {vvar: solve_line(hyp - j, vvar)}
        while gb != [1] or 0 in subs(ls, [*Positive, *NotNull, *Fam]):
            j += 1
            lF = [*Equations, *[_p + rr() for _p in Fam], hyp - j]
            lhyp = [randpoly(vars, degree=1, dense=True)
                    for k in range(len(vars) - len(lF) + 1)]
            gb = MSolveGroebnerLM([*lhyp, *lF], 0, vars, opts)
            ls = {vvar: solve_line(hyp - j, vvar)}
        return hyp, [j]