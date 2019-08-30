# tensorflow_probability_ukf

Tensorflow Probability implementation of 
- Linear Gaussian State Space Model (LGSSM).
- Unscented Kalman Filter for Nonlinear State Space Model (UKFSM).

Parameter estimation in LGSSM is done via direct optimization of the likelihood function using TF automatic differentiation.

Parameter estimation in the UKFSM is achieved via the expectation maximization algorithm for the special case when 
model functions are linear combinations of the parameters, see [2] for details.

For more information on the UKF and EM see

[1] "The unscented Kalman filter for nonlinear estimation", https://ieeexplore.ieee.org/document/882463
[2] "Sigma-Point Filtering and Smoothing Based Parameter Estimation in Nonlinear Dynamic Systems", https://arxiv.org/abs/1504.06173

