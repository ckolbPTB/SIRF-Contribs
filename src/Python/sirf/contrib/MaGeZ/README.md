# MaGeZ contribution to the PETRIC reconstruction challenge

This repository contains the algorithm [ALG1](https://github.com/SyneRBI/PETRIC-MaGeZ/blob/ALG1/main.py) developed by the MaGeZ team submitted to
the [2024 PETRIC reconstruction challenge](https://github.com/SyneRBI/PETRIC) (with minor edits by @paskino to reduce the use of video RAM) and
published in [Fast PET Reconstruction with Variance Reduction and Prior-Aware Preconditioning](https://doi.org/10.3389/fnume.2025.1641215).

## Authors

- Matthias Ehrhardt, University of Bath, United Kingdom
- Georg Schramm, KU Leuven, Belgium
- Zeljko Kereta, University College London, United Kingdom


### ALG1 description

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

### Further info on MaGeZ submission

Please refer to the original repository by the MaGeZ team for further information and notes on the submitted algorithms: [ALG1](https://github.com/SyneRBI/PETRIC-MaGeZ/tree/ALG1), [ALG2](https://github.com/SyneRBI/PETRIC-MaGeZ/tree/ALG2), [ALG3](https://github.com/SyneRBI/PETRIC-MaGeZ/tree/ALG3) .
