# Original results

**Wide Residual Networks**
by Sergey Zagoruyko and Nikos Komodakis

Paper: https://arxiv.org/pdf/1605.07146.pdf

## CIFAR-10

| Model               | Their | Our (JAX) | Our (PYT) | Our (TF2) | Parameters |
|---------------------|-------|------------|-----------|-----------|------------|
| WRN-52-1            | 93.57 |            |           |           |            |
| WRN-52-1 (dropout)  | 93.72 |            |           |           |            |
| WRN-16-4            | 94.98 |            |           |           |            |
| WRN-16-4 (dropout)  | 94.76 |            |           |           |            |
| WRN-40-4            | 95.47 |            |           |           |            |
| WRN-16-8            | 95.73 |            |           |           | 10,962,266 |
| WRN-28-10           | 96.00 |            |           |           |            |
| WRN-28-10 (dropout) | 96.76 |            |           |           |            |

* Their results are obtained by computing median over 5 runs
* The number of parameters are calculated including all the learnable parameters.
For batch normalization, this means we exclude running mean and running std, but include learned mean and std.
* Using normalized data (opposed to ZCA whitened variant).

