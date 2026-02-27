# quantum-computing-explorations

A couple of notebooks I wrote to get hands-on with encoding computational problems into quantum systems. The goal was to use all the theoretical learning and implement things : build the Hamiltonians, run the variational loops, and see what comes out.

## Notebooks

### [Protein Folding as Energy Minimisation](protein_folding_energy.ipynb)

Protein folding cast as a ground state problem. I use the HP lattice model to define an energy landscape over all possible folds of a short chain, then encode it as a diagonal Hamiltonian and solve it with VQE using Qiskit. The variational solver puts over 99% of the measurement probability on ground state configurations of a 6-qubit system, matching the brute-force answer.

### [Variational Quantum Classifier](variational_quantum_classifier.ipynb)

A parameterised 2-qubit circuit trained to classify structured synthetic data with a non-trivial XOR-like label rule. Uses data re-uploading (features encoded in every layer with trainable weights) and tests compositional generalisation - the circuit scores 3/3 on held-out feature combinations it never saw during training. Removing the entangling gates collapses performance entirely (training loss stuck at ln(2), accuracy at chance), which suggests the circuit genuinely relies on quantum correlations to learn the compositional structure.

## Setup

```
pip install numpy scipy matplotlib qiskit qiskit-aer pylatexenc
```
