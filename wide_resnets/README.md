# Original results

**Wide Residual Networks**
by Sergey Zagoruyko and Nikos Komodakis

Paper: https://arxiv.org/pdf/1605.07146.pdf

## CIFAR-10

| Model               | Their | Their code (PYT) |                 Our (JAX)                 | Our (PYT) | Our (TF2) | Parameters |
|---------------------|:-----:|:----------------:|:-----------------------------------------:|:---------:|:---------:|:----------:|
| WRN-52-1            | 93.57 |                  |                                           |           |           |            |
| WRN-52-1 (dropout)  | 93.72 |                  |                                           |           |           |            |
| WRN-16-4            | 94.98 |   95.01 95.06    |                                           |           |           |            |
| WRN-16-4 (dropout)  | 94.76 |                  |                                           |           |           | 2,748,890  |
| WRN-40-4            | 95.47 |   95.64 95.76    | 0.9566 0.9574 0.9562 0.9569 0.9539 0.9559 |           |           | 8,949,210  |
| WRN-16-8            | 95.73 |   95.63 95.65    |                                           |           |           | 10,962,266 |
| WRN-28-10           | 96.00 |   96.03 96.12    |                                           |           |           |            |
| WRN-28-10 (dropout) | 96.76 |                  |                                           |           |           |            |

* Final accuracy is accuracy from the last epoch, not the best one
* Their results are obtained by computing median over 5 runs
* The number of parameters are calculated including all the learnable parameters.
  For batch normalization, this means we exclude running mean and running std, but include learned mean and std.
* Using normalized data (opposed to ZCA whitened variant)

