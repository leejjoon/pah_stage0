# Analytic harmonic frequencies and IR intensities for open-shell (UKS/UHF) molecules with GPU4PySCF.
# GPU4PySCF's properties/ir.py (eval_ir_freq_intensity) asserts RHF/RKS. This is the same algorithm for UHF/UKS:
# the analytic Hessian gives the normal modes; the atomic polar tensors (dipole derivatives) come from the CPHF
# orbital response mo1 that the Hessian code solves, summed over both spins. Closed-shell code: dm1 = C mo1 Cocc^T
# with weight 4 (2 electrons per orbital x 2 for the symmetric derivative); open-shell: alpha and beta each with
# weight 2. Validated by d1/uks_ir_test.py: UKS on a closed-shell molecule reproduces eval_ir_freq_intensity, and
# the naphthalene cation reproduces PAHdb.
from functools import reduce
import numpy as np
import cupy
from pyscf.data import elements, nist
from pyscf.hessian import thermo
from scipy.constants import physical_constants
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import contract

LINDEP_THRESHOLD = 1e-7


def eval_ir_freq_intensity_uks(mf, hessian_obj):
    """(frequencies in cm^-1, intensities in km/mol, hessian) for a converged UKS/UHF object on the GPU."""
    log = logger.new_logger(hessian_obj, mf.mol.verbose)
    hessian = hessian_obj.kernel()
    mol = mf.mol; natm = mol.natm; nao = mol.nao
    hartree_kj = nist.HARTREE2J * 1e3
    unit2cm = ((hartree_kj * nist.AVOGADRO) ** .5 / (nist.BOHR * 1e-10) / (2 * np.pi * nist.LIGHT_SPEED_SI) * 1e-2)
    mass = cupy.array([elements.MASSES[mol.atom_charges()[i]] for i in range(natm)])
    hm = contract('ijkl,i->ijkl', cupy.array(hessian), 1 / cupy.sqrt(mass))
    hm = contract('ijkl,j->ijkl', hm, 1 / cupy.sqrt(mass))
    TR = thermo._get_TR(mass.get(), mol.atom_coords())
    rotor = thermo._get_rotor_type(thermo.rotation_const(mass.get(), mol.atom_coords()))
    TRspace = [TR[:3]] + ([TR[3:5]] if rotor == "LINEAR" else [] if rotor == "ATOM" else [TR[3:]])
    TRspace = cupy.asarray(np.vstack(TRspace))
    q, _ = cupy.linalg.qr(TRspace.T)
    P = cupy.eye(natm * 3) - q.dot(q.T)
    w, v = cupy.linalg.eigh(P); bvec = v[:, w > LINDEP_THRESHOLD]
    h = reduce(cupy.dot, (bvec.T, hm.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm), bvec))
    e, mode = cupy.linalg.eigh(h); mode = bvec.dot(mode)
    c = contract('ixn,i->ixn', mode.reshape(natm, 3, -1), 1 / np.sqrt(mass)).reshape(3 * natm, -1)
    freq = cupy.sign(e) * cupy.sqrt(cupy.abs(e)) * unit2cm

    mo_coeff = cupy.asarray(mf.mo_coeff); mo_occ = cupy.asarray(mf.mo_occ); mo_energy = cupy.asarray(mf.mo_energy)
    dm0 = cupy.asarray(mf.make_rdm1()); dm0 = dm0[0] + dm0[1]
    atmlst = range(natm)
    h1 = hessian_obj.make_h1(mo_coeff, mo_occ, None, atmlst)
    fx = hessian_obj.gen_vind(mo_coeff, mo_occ)
    (mo1a, mo1b), _ = hessian_obj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1, fx, atmlst, hessian_obj.max_memory, log)
    mo1 = (cupy.asarray(mo1a), cupy.asarray(mo1b))
    mocc = [mo_coeff[s][:, mo_occ[s] > 0] for s in (0, 1)]

    tmp = cupy.empty((3, 3, natm))
    aoslices = mol.aoslice_by_atom()
    with mol.with_common_orig((0, 0, 0)):
        hmuao = cupy.array(mol.intor('int1e_r'))
        hmuao11 = cupy.array(-mol.intor('int1e_irp').reshape(3, 3, nao, nao))
    for ia in atmlst:
        p0, p1 = aoslices[ia][2:]
        h11ao = cupy.zeros((3, 3, nao, nao))
        h11ao[:, :, :, p0:p1] += hmuao11[:, :, :, p0:p1]
        h11ao[:, :, p0:p1] += hmuao11[:, :, :, p0:p1].transpose(0, 1, 3, 2)
        tmp[:, :, ia] = 0
        for s in (0, 1):
            t0 = contract('ypi,vi->ypv', mo1[s][ia], mocc[s])
            dm1 = contract('ypv,up->yuv', t0, mo_coeff[s])
            tmp[:, :, ia] -= contract('xuv,yuv->xy', hmuao, dm1) * 2
        tmp[:, :, ia] -= contract('xyuv,vu->xy', h11ao, dm0)
        tmp[:, :, ia] += mol.atom_charge(ia) * cupy.eye(3)
    alpha = physical_constants["fine-structure constant"][0]; amu = physical_constants["atomic mass constant"][0]
    m_e = physical_constants["electron mass"][0]; N_A = physical_constants["Avogadro constant"][0]
    a_0 = physical_constants["Bohr radius"][0]
    unit_kmmol = alpha ** 2 * (1e-3 / amu) * m_e * N_A * np.pi * a_0 / 3
    inten = contract('xym,myn->xn', tmp, c.reshape(natm, 3, -1))
    inten = contract('xn,xn->n', inten, inten) * unit_kmmol
    return freq.get(), inten.get(), hessian
