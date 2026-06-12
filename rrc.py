
from raglib import *
from sage.all import Ideal, matrix, PolynomialRing, QQ, qepcad, show, lcm, prod, SR

def GroebnerBasis(Fs, Rp, Rq, Rf):
    Fs = [Rf(f) for f in Fs]
    Gb = Ideal(Fs).groebner_basis()

    nvars = len(Rq.gens())
    vsB = list(Rf.gens()[:nvars])
    Rvars = PolynomialRing(QQ, vsB)
    FsB = [Rvars(f.subs({Rf(p):1 for p in Rp.gens()})) for f in Fs]
    
    B = [Rf(fb) for fb in Ideal(FsB).normal_basis()]
    return Gb, B[::-1]

def ReduceGB(Gb, Rp, Rq, Rf):
    gbr = list(Gb)
    i=0
    while i < len(gbr):
        fb = gbr[i]
        if Rq(fb) in Ideal(gbr[:i] + gbr[i+1:]).change_ring(Rq):
            gbr.pop(i)
        else:
            i+=1
    return gbr

def XMatrix(v, Gbr, B, Rp, Rq, Rf):
    mtx = []
    for b in B:
        tmp = Rq(v*b)
        red = Ideal(Gbr).change_ring(Rq).reduce(tmp)
        coeffs_by_term = {term:Rp(coeff) for coeff, term in red}
        
        mtx.append([coeffs_by_term.get(Rq(b),0) for b in B])
    return matrix(mtx).transpose()

def BMatrices(XMatrices, B, Rq):
    vs = list(Rq.gens())
    matdict = {
        tuple(0 for _ in vs): matrix.identity(len(B))
    }
    def recurse(degrees):
        if degrees in matdict:
            return matdict[degrees]
        else:
            for i in range(len(vs)):
                if degrees[i] > 0:
                    new_degrees = list(degrees)
                    new_degrees[i]-=1
                    matdict[degrees] = XMatrices[i] * recurse(tuple(new_degrees))
                    return matdict[degrees]

    res = [recurse(Rq(fb).degrees()) for fb in B]
    show([Rq(fb).degrees() for fb in B])
    show(matdict)
    return res

def TraceComputing(BMatrices):
    N = len(BMatrices)
    MTrace = [
        [0 for _ in range(N)]
        for _ in range(N)
    ]
    for i in range(N):
        for j in range(i, N):
            tr = (BMatrices[i]*BMatrices[j]).trace()
            MTrace[i][j] = tr
            MTrace[j][i] = tr

    return matrix(MTrace)

def RemoveDenominators():
    pass

def RandomShift(Mx):
    return Mx # TODO

def LeadingCoefficient(f, Rp, Rq):
    return Rp(list(Rq(f))[0][0])

def SquareFree(f):
    return prod( [ factor for (factor,power) in f.factor() ] )

def DRLHermite(Fs, Rp, Rq, Rf):
    Id = Ideal(Fs)
    Gb, B = GroebnerBasis(Fs, Rp, Rq, Rf)
    Gbr = ReduceGB(Gb, Rp, Rq, Rf)
    Mx = [XMatrix(Rf(v), Gbr, B, Rp, Rq, Rf) for v in Rq.gens()]
    Bxs = BMatrices(Mx, B, Rq)
    Mtr = TraceComputing(Bxs)
    winf = SquareFree(lcm([LeadingCoefficient(f, Rp, Rq) for f in Gb]))

    return Mtr, winf

def LeadingPrincipleMinors(Mx):
    res = []
    for i in range(len(Mx.rows())):
        res.append(Mx[:i+1,:i+1].determinant())

    return res

def SamplePoints(params, eqns):
    qf = qepcad(SR(prod(eqns)) != 0, interact=True)
    qf.go(), qf.go(), qf.go()

    cells = qf.make_cells(qf.d_true_cells())

    return [c.sample_point() for c in cells]

def NumberOfRealSolutions(params, sp, Mx):
    LPMs = LeadingPrincipleMinors(Mx)
    subs_dict = {v:p for v,p in zip(params, sp)}
    LPMs_subs = [f.subs(subs_dict) for f in LPMs]
    if any([val == 0 for val in LPMs_subs]):
        print("ERROR")
    sign_changes = 0
    signs = [1 if val > 0 else -1 for val in LPMs_subs]
    for i in range(1, len(LPMs_subs)):
        if LPMs_subs[i] * LPMs_subs[i-1] < 0:
            sign_changes += 1


    return signs, len(Mx.rows()) - 2*sign_changes