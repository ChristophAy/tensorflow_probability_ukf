import numpy as np
import os
import pdb
import matplotlib.pylab as plt
from scipy.stats import norm
import seaborn as sns
import pandas as pd
from scipy.signal import lfilter
import tensorflow as tf
import random
import tensorflow_probability as tfp
tfd = tfp.distributions
tfl = tf.linalg
from copy import deepcopy

class lgss_model(object):

    def __init__(self, y_ph, a0, P0, parameters, update_fun=None):

        self.y_ph = y_ph
        self.a0 = a0
        self.P0 = P0
        self.Nt = self.y_ph.shape[0]
        self.parameters = parameters

        if update_fun is None:
            self.update_parameters = self.update_parameters_default
        else:
            self.update_parameters = update_fun

        self.make_initial_kalman_state()

        self.initial_smoother_state = (
            tf.zeros_like(self.a0), 
            tf.zeros_like(self.P0),
            self.a0,
            self.P0
        )

    def make_initial_kalman_state(self):

        # initialize Kalman state
        Ht, Qt, Zt, Tt = self.parameters
        H, Q, Z, T = Ht(0), Qt(0), Zt(0), Tt(0)
        P_prior = self.P0

        F = tf.matmul(
                Z, tf.matmul(
                    P_prior, Z, transpose_b=True
                )
            ) + H

        Finv = tf.linalg.inv(F)
        K = tf.matmul(P_prior, 
                    tf.matmul(Z, Finv, transpose_a=True)
                    )

        v = tf.zeros_like(
            tf.matmul(Z, self.a0)
        )
        
        t0 = tf.constant(0)
        ll0 = tf.constant([[0.]])

        self.initial_kalman_state = (
            self.a0, self.P0, ll0, t0,
            Finv, K, v, self.a0, self.P0
        )


    def get_kalman_filter_step(self, params):
    
        def _kalman_filter_step(kalman_state, y):

            '''
            Performs one Kalman filter step
            '''
        
            Ht, Qt, Zt, Tt = params
            
            a_prior, P_prior, ll, t, _, _, _, _, _ = kalman_state
            H, Q, Z, T = Ht(t), Qt(t), Zt(t), Tt(t)

            F = tf.matmul(
                Z, tf.matmul(
                    P_prior, Z, transpose_b=True
                )
            ) + H
            Finv = tf.linalg.inv(F)
            v = y - tf.matmul(Z, a_prior)
            K = tf.matmul(P_prior, 
                        tf.matmul(Z, Finv, transpose_a=True)
                        )
            apo = a_prior + tf.matmul(K,v)
            Ppo = P_prior - tf.matmul(K, tf.matmul(Z, P_prior))
            apr = tf.matmul(T, apo)
            Ppr = tf.matmul(
                T, tf.matmul(Ppo, T, transpose_b=True)
            ) + Q

            ll = - 0.5 * (
                tf.linalg.logdet(F) + 
                tf.matmul(v, tf.matmul(Finv, v), transpose_a=True)
            )

            t+= 1
                
            return (apr, Ppr, ll, t, Finv, K, v, a_prior, P_prior)
        
        return _kalman_filter_step

    def run_kalman_filter(self, params):
        
        _kalman_filter_step = self.get_kalman_filter_step(params)
        
        self.filtered = tf.scan(
            _kalman_filter_step, self.y_ph, self.initial_kalman_state
        )

        return self.filtered

    def get_kalman_smoother_step(self, params):

        def _kalman_smoother_step(smoother_state, filter_state):

            Ht, Qt, Zt, Tt = params

            _, _, _, t, Finv, K, v, apr, Ppr = filter_state
            t =  t-1
            H, Q, Z, T = Ht(t), Qt(t), Zt(t), Tt(t)

            r, N, _, _ = smoother_state

            Kp = tf.matmul(T, K)
            L = T - tf.matmul(Kp, Z)

            r_m = tf.matmul(
                Z, tf.matmul(
                    Finv, v
                ), transpose_a=True
            ) + tf.matmul(
                L, r, transpose_a=True
            )

            N_m = tf.matmul(
                Z, tf.matmul(
                    Finv, Z
                ), transpose_a=True
            ) + tf. matmul(
                L, tf.matmul(
                    N, L
                ), transpose_a=True
            )

            a_smooth = apr + tf.matmul(
                Ppr, r_m
            )

            V_smooth = Ppr - tf.matmul(
                Ppr, tf.matmul(
                    N_m, Ppr
                )
            )

            return (r_m, N_m, a_smooth, V_smooth)

        return _kalman_smoother_step
            
    def run_kalman_smoother(self, params):

        _ = self.run_kalman_filter(params)
        
        _kalman_smoother_step = self.get_kalman_smoother_step(params)
        
        self.smoothed = tf.scan(
            _kalman_smoother_step, self.filtered,
            self.initial_smoother_state, reverse=True
        )

        return self.smoothed

    def update_parameters_default(self, params):

        # This is just a default option

        _params = self.parameters
        _params[0] = tf.exp(params)

        return _params

    def log_prob_eager(self, params):
        
        with tf.GradientTape() as tape:
            
            tape.watch(params)
            print(params)
            _params = self.update_parameters(params)

            _ = self.run_kalman_filter(_params)

            loss = - tf.reduce_sum(
                self.filtered[2]
                )

        grad = tape.gradient(loss, params)

        return loss, grad

    def log_prob(self, params):

        _params = self.update_parameters(params)

        _ = self.run_kalman_filter(_params)

        loss = - tf.reduce_sum(
            self.filtered[2]
            )

        grad = tf.gradients(loss, params)[0]

        return loss, grad

    def approx_second_deriv(self, params_h, params_l, dx):
        
        _, grad_h = self.log_prob(params_h)
        _, grad_l = self.log_prob(params_l)

        d2Ldx2 = (grad_h - grad_l) / (2 * dx)

        return d2Ldx2

    def standard_errors(self, sess, fd, params, dx):

        sds = []

        for i, pa in enumerate(params):
            param_h = deepcopy(params)
            param_l = deepcopy(params)
            param_h[i] += dx
            param_l[i] -= dx

            d2Ldx2 = sess.run(
                self.approx_second_deriv(
                    tf.constant(param_h),
                    tf.constant(param_l), dx),
                    feed_dict = fd
            )

            sd = 1. / np.sqrt(d2Ldx2[i])
            sds.append((pa, sd))

        return sds

    def fit(self, sess, fd, start, tolerance=1e-5,
            dx=1e-6, inital_inverse_hessian=None):

        optim_results = tfp.optimizer.bfgs_minimize(
                self.log_prob, initial_position=start,
                tolerance=tolerance,
                initial_inverse_hessian_estimate=inital_inverse_hessian)

        results = sess.run(optim_results, feed_dict=fd)

        assert(results.converged)
        print ("Function evaluations: %d" % results.num_objective_evaluations)

        sds = self.standard_errors(sess, fd, results.position, dx)

        return results, sds