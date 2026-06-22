# This file is a SageMath translation of RAGlib 
# (Real Algebraic Geometry Library).
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
# Mohab Safey El Din (original author, Maple)
# Andrew Luo (Sage translator)

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
