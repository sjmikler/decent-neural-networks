# 10.12.2022

Experimenting with WRN-16-8. The original model should reach 95.73% accuracy as a median of 5 runs.

I get less -- around 95.5-95.6%.

I try to use L2 regularization with alpha=2e-4 instead of weight decay with alpha=5e-4,
because I remember it worked better. I run the Jax version.

I'll run all the versions from the authors' repo (PyTorch version).
Results:

