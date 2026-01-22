# MaGeZ contribution to the PETRIC reconstruction challenge

This repository contains algorithms developed by the MaGeZ team submitted to
the [2024 PETRIC reconstruction challenge](https://github.com/SyneRBI/PETRIC) and
published in the preprint [Fast PET Reconstruction with Variance Reduction and Prior-Aware Preconditioning](https://arxiv.org/abs/2506.04976).

## Authors

- Matthias Ehrhardt, Univeristy of Bath, United Kingdom
- Georg Schramm, KU Leuven, Belgium
- Zeljko Kereta, Univeristy College London, United Kingdom

## Simulation results related to 2024 PETRIC and our paper

To reproduce all simulation results checkout the tag `2024_paper_simulation_results`
and have a look at the [Readme.md in the simulation_src folder](https://github.com/SyneRBI/PETRIC-MaGeZ/blob/main/simulation_src/README.md).

## Algorithms sumbitted to the 2024 PETRIC challenge

**To reproduce the PETRIC results of our three submitted algorithms
(ALG1, ALG2, ALG3), please checkout the respective tag.**

### ALG1 (main branch, ALG1 tag)

Better-preconditioned SVRG based on the works of:
- R. Twyman et al., "An Investigation of Stochastic Variance Reduction Algorithms for Relative Difference Penalized 3D PET Image Reconstruction," in IEEE TMI, 4, 2023, [link](https://ieeexplore.ieee.org/document/9872020)
- J. Nuyts et al., ""A concave prior penalizing relative differences for maximum-a-posteriori reconstruction in emission tomography," in IEEE TNS, 49, 2022, [link](https://ieeexplore.ieee.org/document/998681)

The update can be summarized as:

$$
x^+ = x + \alpha P \tilde{\nabla} f(x)
$$

where $\tilde{\nabla} f(x)$ is the SVRG stochastic gradient of a subset and
$P$ is a diagonal preconditioner calculated as the harmonic mean of
$x / A^T 1$ and the inverse of diagonal of the Hessian of the RDP, as proposed
in work of Nuyts et al.
The scalar step size $\alpha$ is set to 3 in the first epochs and then decreased
to 1 and finally to 0.5 in late epochs.

### ALG2 (ALG2 branch, ALG2 tag)

Similar to ALG1 but:
The stepsize is set to 2.2 in the first epoch, and is then computed using the Barzilai-Borwein rule as described in Algorithm 2 in [https://arxiv.org/abs/1605.04131](https://arxiv.org/abs/1605.04131),
with m = 1 but using the short stepsize (see [https://en.wikipedia.org/wiki/Barzilai-Borwein_method]([https://en.wikipedia.org/wiki/Barzilai-Borwein_method])) adapted to preconditioned gradient ascent methods

### ALG3 (ALG3 branch, ALG3 tag)

Using the same setup as in ALG2 but with two minor differences.
First, we use a slightly smaller number of subsets.
Second, we use a non-stochastic selection of subset indices by considering the cyclic group corresponding to the given number of subsets, finding all of its generators (i.e. the set of all coprimes of the number of subsets that are smaller), and then creating indices by consider specific generators at a time.
