def PointsPerComponents(eqs, pos, ineqs, opts={}):
    _l = None
    newopts = None

    is_empty = opts.get("isempty", False)
    
    if not is_empty:
        return SemiAlgebraicSolve(eqs, pos, ineqs, opts);
    else:
        newopts = {
            k: 0 if k == "isempty" else v
            for k, v in opts.items()
        }
        return SemiAlgebraicSolve(eqs, pos, ineqs, newopts);

def HasRealSolutions(eqs, pos, ineqs, opts={}):
    _l = None
    newopts = None

    is_empty = opts.get("isempty", False)
    
    if not is_empty:
        opts |= {"isempty":1}
        return SemiAlgebraicSolve(eqs, pos, ineqs, opts)
    else:
        newopts = {
            k: 1 if k == "isempty" else v
            for k, v in opts.items()
        }
        return SemiAlgebraicSolve(eqs, pos, ineqs, newopts)

def DRLHermite(Fs, Rp, Rq, Rf):
    Id = Ideal(Fs)
    Gb, B = GroebnerBasis(Fs, Rp, Rq, Rf)
    Gbr = ReduceGB(Gb, Rp, Rq, Rf)
    Mx = [XMatrix(Rf(v), Gbr, B, Rp, Rq, Rf) for v in Rq.gens()]
    Bxs = BMatrices(Mx, B, Rq)
    Mtr = TraceComputing(Bxs)
    winf = SquareFree(lcm([LeadingCoefficient(f, Rp, Rq) for f in Gb]))

    return Mtr, winf