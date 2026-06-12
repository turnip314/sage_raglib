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
