# This file is part of msolve.
#
# msolve is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# msolve is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with msolve.  If not, see <https://www.gnu.org/licenses/>
#
# Authors:
# Christian Eder
# Jorge Garcia Fontan
# Huu Phuoc Le
# Mohab Safey El Din
# Bruno Salvy

import os
import tempfile
import subprocess
import platform
import random
import string
import math
from fractions import Fraction
from helpers import isprime, debug, extend_ring, convert_to_ring
from sage.all import Ideal
from sage.misc.sage_eval import sage_eval
from sage.features.msolve import msolve
from sage.sets.primes import Primes
from sage.symbolic.relation import solve
from sage.arith.misc import gcd
from sage.all import mul, log, floor, binomial, ceil, next_prime
from sage.matrix.constructor import Matrix
from sage.symbolic.ring import SR
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.qqbar import AA, QQbar
from sage.rings.integer import Integer
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.term_order import TermOrder
from sage.rings.continued_fraction import continued_fraction

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def GetSystem():
    sys_str = platform.system()
    if sys_str == "Darwin":
        return "macOS"
    elif sys_str == "Linux":
        return "Linux"
    else:
        return "Windows"


def _random_string(n=8):
    return ''.join(random.choices(string.ascii_lowercase, k=n))


def ReadPolynomial(L, var):
    """
    L = [deg, [c0, c1, ..., cdeg]]  (1-indexed in Maple: L[1]=deg, L[2]=coefflist)
    Translated: L[0] = deg+1 count, L[1] = coeff list.
    add(L[2][i]*var^(i-1), i=1..L[1]+1)  -> sum over 0-based index
    """
    n = L[0]  # L[1] in Maple
    coeffs = L[1]  # L[2] in Maple
    return sum(coeffs[i] * var**i for i in range(n + 1))


def ToMSolve(F, char, vars, fname):
    with open(fname, 'w') as fd:
        # Write variable names separated by commas
        for i in range(len(vars) - 1):
            fd.write(f"{vars[i]}, ")
        fd.write(f"{vars[-1]} \n")
        fd.write(f"{char}\n")

        if char == 0:
            F2 = [f for f in F if f != 0]
            for i in range(len(F2) - 1):
                fd.write(f"{F2[i]},\n")
            fd.write(f"{F2[-1]}\n")
        else:
            F2 = [f for f in F if f != 0]
            for i in range(len(F2) - 1):
                fd.write(f"{F2[i]},\n")
            fd.write(f"{F2[-1]}\n")

    # Join continuation lines with sed (macOS vs Linux syntax)
    if GetSystem() == "macOS":
        str_cmd = f"sed -i '' -e ':a' -e 'N' -e '$!ba' -e 's/\\\\\\n//g' {fname}"
    else:  # Linux
        str_cmd = f"sed -i -e ':a' -e 'N' -e '$!ba' -e 's/\\\\\\n//g' {fname}"
    os.system(str_cmd)


def GetOptions(opts):
    """
    opts: dict mapping option names to values.
    Returns: (str_cmd, fname1, fname2, verb, output)
    """
    # verb
    verb = opts.get("verb", 0)
    if isinstance(verb, int):
        verb = min(verb, 2)
        if verb < 2:
            verb = 0
    else:
        verb = 0

    # gb
    gb_val = opts.get("gb", 0)
    gb = gb_val if (isinstance(gb_val, int) and gb_val > 0) else 0

    # elim
    elim_val = opts.get("elim", 0)
    elim = elim_val if (isinstance(elim_val, int) and elim_val > 0) else 0

    # trunc
    trunc_val = opts.get("trunc", -1)
    truncate = trunc_val if (isinstance(trunc_val, int) and trunc_val > 0) else -1

    # output
    output = opts.get("output", 0)
    if not isinstance(output, int):
        output = 0

    # nthreads
    nthreads = opts.get("nthreads", 1)
    if not isinstance(nthreads, int):
        nthreads = 1

    # linalg
    linalg = opts.get("linalg", 2)
    if not isinstance(linalg, int):
        linalg = 2

    # file_dir
    if "file_dir" in opts:
        file_dir = opts["file_dir"]
        if not isinstance(file_dir, str):
            print("Error in format options")
            file_dir = "/tmp/"
    else:
        file_dir = "/tmp/"

    # fname1 (input file)
    if "file_in" in opts:
        val = opts["file_in"]
        if isinstance(val, str):
            fname1 = os.path.join(file_dir, val)
        else:
            print("Error in format options")
            fname1 = os.path.join(file_dir, _random_string(8) + ".ms")
    else:
        fname1 = os.path.join(file_dir, _random_string(8) + ".ms")

    # fname2 (output file)
    if "file_out" in opts:
        val = opts["file_out"]
        if isinstance(val, str):
            fname2 = os.path.join(file_dir, val)
        else:
            print("Error in format options")
            fname2 = os.path.join(file_dir, _random_string(8) + ".ms")
    else:
        fname2 = os.path.join(file_dir, _random_string(8) + ".ms")

    # param
    param = opts.get("param", 0)
    if not isinstance(param, int):
        param = 0

    msolve_path = msolve().absolute_filename()
    # Build the msolve command string
    str_cmd = f"{msolve_path} -v {verb}"
    if not (gb == 0 or param != 0):
        str_cmd += f" -g {gb}"
    if elim != 0:
        str_cmd += f" -e {elim}"
    if param != 0:
        str_cmd += f" -P {param}"
    if truncate != -1:
        str_cmd += f" -N {truncate}"
    if nthreads != 1:
        str_cmd += f" -t {nthreads}"
    if linalg != 2:
        str_cmd += f" -l {linalg}"
    str_cmd += f" -f {fname1} -o {fname2}"

    return str_cmd, fname1, fname2, verb, output


def RemoveFiles(fname1, fname2):
    os.system(f"rm {fname2}")
    os.system(f"rm {fname1}")


def CallMSolve(F, fc, vars, opts):
    str_cmd, fname1, fname2, verb, output = GetOptions(opts)

    nvars = len(vars)
    # Rename variables to internal names _xxi
    xx = list(SR.var('xx_', nvars))
    subs_in  = {vars[i]: xx[i] for i in range(nvars)}
    subs_out = {xx[i]: vars[i] for i in range(nvars)}

    F_renamed = [f.subs(subs_in) for f in F]
    ToMSolve(F_renamed, fc, xx, fname1)

    # Set float precision
    prec = 64  # default (Digits <= 10 in Maple)

    if fc == 0:
        str_cmd = str_cmd + f" -p {prec}"

    try:
        if verb == 0:
            os.system(str_cmd)
        else:
            os.system(str_cmd)
        with open(fname2, 'r') as f:
            raw = f.read()[:-2]
        results_renamed = sage_eval(raw)
        # Substitute internal variable names back
        results = _subs_results(results_renamed, subs_out)
    except Exception as e:
        # Write a debug file analogous to /tmp/bug-call-msolve.mpl
        with open("/tmp/bug-call-msolve.py", 'w') as fd:
            fd.write(f"F = {F}\nfc = {fc}\nvars = {vars}\nopts = {opts}\n")
        print(str_cmd)
        raise RuntimeError(
            "There has been an issue in msolve computation (see /tmp/bug-call-msolve.py)"
        ) from e

    RemoveFiles(fname1, fname2)
    return results, output


def _subs_results(obj, subs_map):
    """Recursively apply a substitution dict to a nested structure."""
    if isinstance(obj, list):
        return [_subs_results(x, subs_map) for x in obj]
    if hasattr(obj, 'subs'):
        return obj.subs(subs_map)
    return obj


def CheckCharacteristic(fc):
    if fc != 0 and not isprime(fc):
        raise ValueError("Field characteristic should be a prime number")
    if fc > 2**31:
        raise ValueError("Field characteristic > 2^31 not supported")
    return fc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def MSolveGroebner(F, fc, vs, opts=None):
    """
    Compute a Groebner basis of the ideal generated by F.

    Input:
      F    : list of polynomials with rational coefficients
      fc   : field characteristic (nonneg int; 0 = rationals)
      vars : list of variables
      opts : dict of options (see comments below)

    Options:
      "mspath"   : path to msolve binary
      "verb"     : verbosity (positive int)
      "file_dir" : directory for intermediate files
      "file_in"  : input filename
      "file_out" : output filename
      "leadmons" : 1 to return only leading monomials
      "elim"     : number of variables to eliminate

    Output:
      []     -> error during computation
      else   -> Groebner basis (grevlex, vars[0] > ... > vars[n-1])
    """
    if opts is None:
        opts = {}
    field_char = CheckCharacteristic(fc)
    R_original = vs[0].parent()

    merged_opts = {**opts, "gb": 2}
    str_cmd, fname1, fname2, verb, output = GetOptions(merged_opts)

    nvars   = len(vs)
    xx      = list(SR.var('xx_', nvars))
    R = PolynomialRing(QQ, xx)
    xx = list(R.gens())
    subs_in  = {vs[i]: xx[i] for i in range(nvars)}
    subs_out = {xx[i]: vs[i] for i in range(nvars)}

    F_renamed = [f.subs(subs_in) for f in F]
    ToMSolve(F_renamed, field_char, xx, fname1)

    try:
        if verb == 0:
            os.system(str_cmd)
        else:
            os.system(str_cmd)
        with open(fname2, 'r') as f:
            raw = f.read()[:-2]
        results_renamed = sage_eval(raw, locals={f'{v}': v for v in xx})
        results = _subs_results(results_renamed, subs_out)
        results = [R_original(r) for r in results]
        RemoveFiles(fname1, fname2)
        return results
    except Exception as e:
        raise RuntimeError("There has been an issue in msolve computation") from e


def MSolveGroebnerLM(F, fc, vs, opts=None):
    """
    Like MSolveGroebner but returns only leading monomials (gb=1).
    """
    if opts is None:
        opts = {}
    field_char = CheckCharacteristic(fc)
    R_original = vs[0].parent()

    merged_opts = {**opts, "gb": 1}
    str_cmd, fname1, fname2, verb, output = GetOptions(merged_opts)

    nvars   = len(vs)
    xx      = list(SR.var('xx_', nvars))
    R = PolynomialRing(QQ, xx)
    xx = list(R.gens())
    subs_in  = {vs[i]: xx[i] for i in range(nvars)}
    subs_out = {xx[i]: vs[i] for i in range(nvars)}

    F_renamed = [f.subs(subs_in) for f in F]
    ToMSolve(F_renamed, field_char, xx, fname1)

    try:
        if verb == 0:
            os.system(str_cmd)
        else:
            os.system(str_cmd)
        with open(fname2, 'r') as f:
            raw = f.read()[:-2]
        results_renamed = sage_eval(raw, locals={f'{v}': v for v in xx})
        results = _subs_results(results_renamed, subs_out)
        results = [R_original(r) for r in results]
        RemoveFiles(fname1, fname2)
        return results
    except Exception as e:
        print(results)
        print(R)
        raise e
        raise RuntimeError("There has been an issue in msolve computation") from e


def get_parametrization(vs, system, fc=0, allow_null=False):
    filename = msolve().absolute_filename()
    msolve_in = tempfile.NamedTemporaryFile(mode="w", encoding="ascii", delete=False)
    command = [filename, "-f", msolve_in.name, "-P", "2"]

    system = list(str(e) for e in system)
    try:
        print(",".join([str(v) for v in vs]), file=msolve_in)
        print(fc, file=msolve_in)
        print(*(pol.replace(" ", "") for pol in system), sep=",\n", file=msolve_in)
        f = open(msolve_in.name)
        msolve_in.close()
        
        msolve_out = subprocess.run(command, capture_output=True, text=True)
    finally:
        os.unlink(msolve_in.name)

    msolve_out.check_returncode()

    result = msolve_out.stdout
    result = sage_eval(result[:-2])

    if allow_null and result[0] == -1:
        return None
    if result[0] != 0:
        print(result)
        raise Exception(
            "Issue with msolve parametrization - system does not have finitely many solutions"
        )

    return result

def MSolveParam(Fs, fc, vs, allow_null=False, opts=None):
    """
    Compute a rational parametrization of the solutions of F.

    Returns [elim_poly, [var_i = expr_i, ...]] where each coordinate
    is expressed as a rational function of the root of elim_poly.
    """
    result = get_parametrization(vs, Fs, allow_null=allow_null)
    if result is None:
        return None

    _, nvars, _, msvars, _, param = result[1]

    # msolve may reorder the variables, so order them back
    Qparams = param[1][2]
    vsExt = [str(v) for v in vs]
    # Check if no new variable was created by msolve
    # If so, the linear_form used is just zd
    # i.e. u_ = zd and Qd = zd * P'(zd)
    if nvars == len(vs):
        Pdz = [0] + [-c for c in param[1][1][1]]
        Qparams.append([[1, Pdz], 1])
    pidx = [msvars.index(v) for v in vsExt]
    Qparams = [Qparams[i] for i in pidx]

    u_ = SR.var('u_')
    if fc == 0:
        R = PolynomialRing(QQ, [u_])
        u_ = R(u_)

    P_coeffs = param[1][0][1]
    P = sum([c * u_**i for (i, c) in enumerate(P_coeffs)])

    Qs = []
    for Q_param in Qparams:
        Q_coeffs = Q_param[0][1]
        c_div = Q_param[1]
        Q = -sum([c * u_**i for (i, c) in enumerate(Q_coeffs)]) / c_div
        Qs.append(Q)

    return P, Qs, [] # TODO - give msolve approxs

def MSolveRealRoots(Fs, vs, G=None, opts=None):
    """
    Compute real roots of the polynomial system F.

    Input:
      F    : list of polynomials with rational coefficients
      vars : list of variables
      G    : list of extra constraint polynomials (can be empty list)
      opts : dict of options

    Options:
      "mspath"   : path to msolve binary
      "verb"     : verbosity
      "file_dir" : directory for intermediate files
      "file_in"  : input filename
      "file_out" : output filename
      "output"   : 1 to return midpoints of isolating intervals

    Output:
      []          -> error
      [1]         -> infinitely many complex solutions
      [-1, []]    -> no real solution
      [0, [sols]] -> finitely many complex solutions; sols is a list of
                     {var: [lo, hi]} dicts (isolating intervals)
    """
    if G is None:
        G = []
    if opts is None:
        opts = {}

    verb = opts.get("verb", 0)
    if not isinstance(verb, int):
        verb = 0

    # If any polynomial is a nonzero constant the system has no solution
    if any(f.degree() == 0 for f in Fs):
        return [-1, []]

    output = opts.get("output", 0)
    if len(G) > 0 and output == 1:
        raise ValueError(
            "mid points cannot be returned with extra constraints: change your options"
        )
    
    R = vs[0].parent()
    Fs = [R(f) for f in Fs if f != 0]
    vs = [R(v) for v in vs]

    merged_opts = {**opts, "param": 1}

    param = MSolveParam(Fs, 0, vs, merged_opts)
    if param is None:
        return [0, []]
    P, Qs, _ = param
    Pd = P.derivative()
    u_ = P.parent().gens()[0]
    sols = []
    for u in P.roots(AA, multiplicities=False):
        sols.append(
            {
                v: [(Qs[i]/Pd).subs({u_:u}), (Qs[i]/Pd).subs({u_:u})] for i, v in enumerate(vs)
            }
        )

    return [0, sols]

def MSolveSat(Fs, Gs, vs, fc=0):
    R = vs[0].parent()
    y_ = SR.var('y_')
    R_ext = extend_ring(R, [y_])
    Fs, Gs, vs, y_ = convert_to_ring(R_ext, Fs, Gs, vs, y_)
    Id = Ideal(R.one())
    for g in Gs:
        sat = MSolveGroebner([g*y_-1, *Fs], fc, [y_, *vs], opts={"elim": 1})
        sat = [R(f) for f in sat]
        Id = Id.intersection(Ideal(sat).change_ring(R))
    return [R(f) for f in Id.gens()]



