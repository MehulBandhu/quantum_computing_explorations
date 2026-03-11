"""
Protein folding on a 3D tetrahedral lattice using VQE.

Folds the Zika virus NS3 helicase P-loop (LHPGAGK, 7 residues) by encoding
backbone moves as qubits and building a diagonal Hamiltonian from
Miyazawa-Jernigan pairwise contact energies. VQE finds the minimum-energy
fold on a simulator, then the same circuit runs on IBM quantum hardware.
The result is compared to the experimental crystal structure (PDB: 5gjb)
via Kabsch alignment and RMSD.

Requirements: numpy, scipy, matplotlib, qiskit, qiskit-aer, qiskit-ibm-runtime
"""

import numpy as np
from itertools import product
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import EfficientSU2
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


# --- lattice and sequence setup ---

# four tetrahedral directions, each of length sqrt(3)
directions = np.array([
    [+1, +1, +1],
    [+1, -1, -1],
    [-1, +1, -1],
    [-1, -1, +1],
])

sequence = "LHPGAGK"
n_residues = len(sequence)
n_free_moves = n_residues - 2  # first residue at origin, first move fixed
n_qubits = 2 * n_free_moves
n_states = 2 ** n_qubits

print(f"Peptide: {sequence} ({n_residues} residues)")
print(f"Free moves: {n_free_moves}, Qubits: {n_qubits}, States: {n_states}")


# --- Miyazawa-Jernigan contact energies (kT units) ---

mj_energies = {
    ('L','L'): -1.08, ('L','H'): -0.60, ('L','P'): -0.62,
    ('L','G'): -0.58, ('L','A'): -0.84, ('L','K'): -0.53,
    ('H','H'): -0.46, ('H','P'): -0.38, ('H','G'): -0.40,
    ('H','A'): -0.43, ('H','K'): -0.29,
    ('P','P'): -0.43, ('P','G'): -0.34, ('P','A'): -0.50,
    ('P','K'): -0.28,
    ('G','G'): -0.31, ('G','A'): -0.40, ('G','K'): -0.27,
    ('A','A'): -0.58, ('A','K'): -0.36,
    ('K','K'): -0.13,
}
for (a, b), e in list(mj_energies.items()):
    mj_energies[(b, a)] = e


def contact_energy(aa_i, aa_j):
    return mj_energies.get((aa_i, aa_j), 0.0)


# --- fold decoding and energy computation ---

def decode_moves(state_idx):
    """extract move indices (0-3) from a basis state integer."""
    moves = []
    for m in range(n_free_moves):
        q_lo = (state_idx >> (2 * m)) & 1
        q_hi = (state_idx >> (2 * m + 1)) & 1
        moves.append(2 * q_hi + q_lo)
    return moves


def compute_positions(moves):
    """lay out residue positions on the tetrahedral lattice."""
    positions = [np.array([0, 0, 0])]
    positions.append(positions[0] + directions[0])  # first move fixed
    for m_idx in moves:
        positions.append(positions[-1] + directions[m_idx])
    return np.array(positions)


def check_collision(positions):
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if np.array_equal(positions[i], positions[j]):
                return True
    return False


def compute_fold_energy(positions, seq):
    """sum MJ contact energies for non-bonded lattice-adjacent pairs."""
    energy = 0.0
    for i in range(len(seq)):
        for j in range(i + 2, len(seq)):
            diff = positions[j] - positions[i]
            if np.sum(diff ** 2) == 3:  # adjacent on tetrahedral lattice
                energy += contact_energy(seq[i], seq[j])
    return energy


# --- enumerate all folds classically ---

collision_penalty = 50.0
energies = np.zeros(n_states)
valid_count = 0
collision_count = 0

for idx in range(n_states):
    moves = decode_moves(idx)
    positions = compute_positions(moves)
    if check_collision(positions):
        energies[idx] = collision_penalty
        collision_count += 1
    else:
        energies[idx] = compute_fold_energy(positions, sequence)
        valid_count += 1

ground_energy = np.min(energies[energies < collision_penalty])
ground_states = [i for i in range(n_states)
                 if abs(energies[i] - ground_energy) < 1e-8]

print(f"Valid folds: {valid_count}, Collisions: {collision_count}")
print(f"Ground state energy: {ground_energy:.4f}")
print(f"Degenerate ground states: {len(ground_states)}")

best_idx = ground_states[0]
best_moves = decode_moves(best_idx)
best_positions = compute_positions(best_moves)


# --- energy landscape ---

valid_energies = energies[energies < collision_penalty]

plt.figure(figsize=(8, 4))
plt.hist(valid_energies, bins=30, edgecolor='black', alpha=0.7, color='#4C72B0')
plt.axvline(x=ground_energy, color='red', linestyle='--',
            label=f'Ground state ({ground_energy:.3f})')
plt.xlabel('Energy (MJ units)')
plt.ylabel('Number of folds')
plt.title(f'Energy landscape: {sequence} on tetrahedral lattice')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- build Pauli Hamiltonian via Walsh-Hadamard transform ---

def walsh_hadamard_transform(eigenvalues, nq):
    """decompose diagonal eigenvalues into weighted Pauli-Z products."""
    n = 2 ** nq
    coefficients = {}
    for s in range(n):
        c = 0.0
        for x in range(n):
            parity = bin(s & x).count('1') % 2
            c += (1 - 2 * parity) * eigenvalues[x]
        c /= n
        if abs(c) > 1e-10:
            pauli_str = ''.join('Z' if (s >> q) & 1 else 'I' for q in range(nq))
            coefficients[pauli_str[::-1]] = c  # reverse for qiskit endianness
    return coefficients


print("building hamiltonian...")
pauli_coeffs = walsh_hadamard_transform(energies, n_qubits)
print(f"pauli terms: {len(pauli_coeffs)}")

pauli_labels = list(pauli_coeffs.keys())
pauli_values = [pauli_coeffs[l] for l in pauli_labels]
h_protein = SparsePauliOp(pauli_labels, pauli_values)

# sanity check against brute force
print("\nverification (ground states):")
h_matrix = h_protein.to_matrix()
for gs_idx in ground_states[:5]:
    e_pauli = np.real(h_matrix[gs_idx, gs_idx])
    match = '✓' if abs(e_pauli - energies[gs_idx]) < 1e-6 else '✗'
    print(f"  |{gs_idx:0{n_qubits}b}> pauli={e_pauli:.4f}  brute={energies[gs_idx]:.4f}  {match}")


# --- VQE ---

ansatz = EfficientSU2(n_qubits, reps=3, entanglement='linear')
print(f"\nansatz: {ansatz.num_parameters} parameters")


def vqe_cost(params, ansatz, hamiltonian):
    qc = ansatz.assign_parameters(params)
    sv = Statevector.from_instruction(qc)
    return np.real(sv.expectation_value(hamiltonian))


best_energy = np.inf
best_params = None
vqe_history = []
n_trials = 20

print(f"running VQE with {n_trials} restarts...")
for trial in range(n_trials):
    x0 = np.random.randn(ansatz.num_parameters) * np.pi
    trial_hist = []

    def callback(p):
        trial_hist.append(vqe_cost(p, ansatz, h_protein))

    result = minimize(
        vqe_cost, x0, args=(ansatz, h_protein),
        method='COBYLA', callback=callback,
        options={'maxiter': 500, 'rhobeg': 0.5}
    )
    vqe_history.extend(trial_hist)

    if result.fun < best_energy:
        best_energy = result.fun
        best_params = result.x
        print(f"  trial {trial}: new best = {best_energy:.4f}")

print(f"\nVQE result: {best_energy:.4f}")
print(f"true ground state: {ground_energy:.4f}")
print(f"relative error: {abs(best_energy - ground_energy) / abs(ground_energy) * 100:.2f}%")


# --- analyse VQE output ---

qc_opt = ansatz.assign_parameters(best_params)
sv_opt = Statevector.from_instruction(qc_opt)
probs = sv_opt.probabilities()

gs_prob = sum(probs[i] for i in ground_states)
print(f"ground state probability: {gs_prob:.4f}")

print("\ntop 5 states:")
for idx in np.argsort(probs)[::-1][:5]:
    e = energies[idx]
    tag = "GROUND STATE" if idx in ground_states else ("COLLISION" if e >= collision_penalty else "")
    print(f"  |{idx:0{n_qubits}b}> prob={probs[idx]:.4f}  E={e:.4f}  {tag}")

vqe_best_idx = np.argmax(probs)
vqe_moves = decode_moves(vqe_best_idx)
vqe_positions = compute_positions(vqe_moves)
vqe_energy = energies[vqe_best_idx]
print(f"\nVQE best fold: moves={vqe_moves}, E={vqe_energy:.4f}")


# --- experimental structure (PDB 5gjb, Cα coordinates) ---

pdb_ca = np.array([
    [-2.829, 16.951,  9.567],  # L
    [-1.034, 14.007,  7.951],  # H
    [ 2.535, 14.532,  6.695],  # P
    [ 2.553, 16.566,  3.473],  # G
    [-1.035, 17.801,  3.935],  # A
    [-0.030, 21.485,  3.994],  # G
    [ 0.042, 22.127,  7.767],  # K
])

print("PDB Cα coordinates loaded")
dists = [np.linalg.norm(pdb_ca[i+1] - pdb_ca[i]) for i in range(len(pdb_ca)-1)]
print(f"consecutive Cα distances: {' '.join(f'{d:.2f}' for d in dists)} Å")


# --- Kabsch alignment and RMSD ---

def kabsch_rmsd(p, q):
    """align p onto q (both Nx3), with scaling from lattice units to angstroms."""
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)

    # scale lattice steps to match Cα spacing
    ca_dist = np.mean([np.linalg.norm(q[i+1] - q[i]) for i in range(len(q)-1)])
    p_c = p_c * (ca_dist / np.sqrt(3))

    h = p_c.T @ q_c
    u, s, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    r = vt.T @ np.diag([1, 1, np.sign(d)]) @ u.T

    p_aligned = (r @ p_c.T).T + q.mean(axis=0)
    rmsd = np.sqrt(np.mean(np.sum((p_aligned - q) ** 2, axis=1)))
    return rmsd, p_aligned


rmsd_vqe, aligned_vqe = kabsch_rmsd(vqe_positions.astype(float), pdb_ca)
print(f"VQE fold RMSD: {rmsd_vqe:.2f} Å")

rmsd_bf, aligned_bf = kabsch_rmsd(best_positions.astype(float), pdb_ca)
print(f"brute-force best RMSD: {rmsd_bf:.2f} Å")

best_rmsd = np.inf
best_aligned = None
for gs_idx in ground_states:
    gs_pos = compute_positions(decode_moves(gs_idx)).astype(float)
    r, a = kabsch_rmsd(gs_pos, pdb_ca)
    if r < best_rmsd:
        best_rmsd = r
        best_aligned = a

print(f"best RMSD across {len(ground_states)} degenerate ground states: {best_rmsd:.2f} Å")
print(f"\nreference: Doga et al. 1.78 Å, AlphaFold2 3.53 Å (same fragment)")


# --- 3D visualisation ---

aa_colours = {
    'L': '#D64541', 'H': '#3498DB', 'P': '#E67E22',
    'G': '#95A5A6', 'A': '#D64541', 'K': '#3498DB',
}


def plot_fold_3d(ax, positions, title, seq):
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    ax.plot(xs, ys, zs, '-', color='black', linewidth=1.5, alpha=0.6)
    for i, (x, y, z) in enumerate(positions):
        ax.scatter(x, y, z, s=120, c=aa_colours.get(seq[i], '#4C72B0'),
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax.text(x, y, z + 0.3, f'{seq[i]}{i}', fontsize=7, ha='center')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')


fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(131, projection='3d')
plot_fold_3d(ax1, pdb_ca, 'Experimental (PDB 5gjb)', sequence)

ax2 = fig.add_subplot(132, projection='3d')
plot_fold_3d(ax2, aligned_vqe, f'VQE simulator\nRMSD = {rmsd_vqe:.2f} Å', sequence)

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(*pdb_ca.T, 'o-', color='#27AE60', linewidth=2, markersize=8, label='Experimental')
ax3.plot(*aligned_vqe.T, 's--', color='#D64541', linewidth=2, markersize=8, label='VQE fold')
for i in range(n_residues):
    ax3.plot([pdb_ca[i,0], aligned_vqe[i,0]],
             [pdb_ca[i,1], aligned_vqe[i,1]],
             [pdb_ca[i,2], aligned_vqe[i,2]], ':', color='gray', alpha=0.5)
ax3.set_title(f'Overlay\nRMSD = {rmsd_vqe:.2f} Å', fontsize=11)
ax3.legend(fontsize=8)
ax3.set_xlabel('X (Å)')
ax3.set_ylabel('Y (Å)')
ax3.set_zlabel('Z (Å)')

plt.tight_layout()
plt.savefig('protein_fold_comparison.png', dpi=200, bbox_inches='tight')
plt.show()


# --- convergence plot ---

plt.figure(figsize=(8, 3))
plt.plot(vqe_history, color='#4C72B0', linewidth=0.8)
plt.axhline(y=ground_energy, color='red', linestyle='--',
            label=f'True ground state ({ground_energy:.4f})')
plt.xlabel('Optimisation step (across all restarts)')
plt.ylabel('Energy')
plt.title(f'VQE convergence — {sequence} ({n_qubits} qubits)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# --- hardware run (requires IBM Quantum account) ---
# save your token once before running:
#   QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="YOUR_TOKEN", overwrite=True)

service = QiskitRuntimeService(channel='ibm_quantum_platform')
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=n_qubits)
print(f"\nbackend: {backend.name} ({backend.num_qubits} qubits)")

qc_hw = ansatz.assign_parameters(best_params)
qc_hw.measure_all()
qc_transpiled = transpile(qc_hw, backend=backend, optimization_level=3)
print(f"transpiled depth: {qc_transpiled.depth()}")
print(f"gate counts: {dict(qc_transpiled.count_ops())}")

sampler = SamplerV2(backend)
job = sampler.run([qc_transpiled], shots=4096)
result = job.result()
counts_hw = result[0].data.meas.get_counts()

best_hw_bitstr = max(counts_hw, key=counts_hw.get)
best_hw_idx = int(best_hw_bitstr, 2)
hw_moves = decode_moves(best_hw_idx)
hw_positions = compute_positions(hw_moves).astype(float)
rmsd_hw, aligned_hw = kabsch_rmsd(hw_positions, pdb_ca)

print(f"hardware best bitstring: {best_hw_bitstr}")
print(f"hardware best energy: {energies[best_hw_idx]:.4f}")
print(f"hardware best RMSD: {rmsd_hw:.2f} Å")
