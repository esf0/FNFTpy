"""
This file is part of FNFTpy.
FNFTpy provides wrapper functions to interact with FNFT,
a library for the numerical computation of nonlinear Fourier transforms.

For FNFTpy to work, a copy of FNFT has to be installed.
For general information, source files and installation of FNFT,
visit FNFT's github page: https://github.com/FastNFT

For information about setup and usage of FNFTpy see README.md or documentation.

FNFTpy is free software; you can redistribute it and/or
modify it under the terms of the version 2 of the GNU General
Public License as published by the Free Software Foundation.

FNFTpy is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Contributors:

Christoph Mahnke, 2018-2021

"""

from FNFTpy import nsev
import numpy as np


def nsev_example(bsf=None, bsl=None, niter=None, Dsub=None, dst=None, cst=None, nf=None, dis=None, ref=None,
                 bound_state_guesses=None, amplitude_scale=1.0, verbose=True):
    """Mimics the C example for calling fnft_nsev."""
    if verbose:
        print("\n\nnsev example")
    D = 256
    tvec = np.linspace(-1, 1, D)
    q = np.zeros(len(tvec), dtype=np.complex128)
    q[:] = amplitude_scale * 2.0 + 0.0j
    M = 8
    Xi1 = -2
    Xi2 = 2
    Xivec = np.linspace(Xi1, Xi2, M)

    # call function
    res = nsev(q, tvec, M=M, Xi1=Xi1, Xi2=Xi2,
               bsf=bsf, bsl=bsl, niter=niter, Dsub=Dsub, dst=dst, cst=cst, nf=nf, dis=dis, ref=ref,
               bound_state_guesses=bound_state_guesses,
               )

    if verbose:
        print("\n----- options used ----")
        print(res['options'])
        print("\n------ results --------")

        print("FNFT return value: %d (should be 0)" % res['return_value'])
        print("continuous spectrum")
        for i in range(len(res['cont_ref'])):
            print("%d :  Xi = %.4f   %.6f  %.6fj" % (
                i, Xivec[i], np.real(res['cont_ref'][i]), np.imag(res['cont_ref'][i])))
        print("discrete spectrum")
        for i in range(len(res['bound_states'])):
            print("%d : %.6f  %.6fj with norming const %.6f  %.6fj" % (i, np.real(res['bound_states'][i]),
                                                                       np.imag(res['bound_states'][i]),
                                                                       np.real(res['disc_norm'][i]),
                                                                       np.imag(res['disc_norm'][i])))
    return res
