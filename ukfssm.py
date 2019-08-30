import numpy as np
import os
import pdb
import tensorflow as tf
import random
import tensorflow_probability as tfp
tfd = tfp.distributions
tfl = tf.linalg
from copy import deepcopy
import tensorflow.contrib.eager as tfe

class ukss_model(object):

    def __init__(self, y_ph, a0, P0, k, parameters, update_fun=None,
                 noise_state_dep=False, regression=False):

        self.y_ph = y_ph
        self.a0 = a0
        self.P0 = P0
        self.Nt = tf.shape(self.y_ph)[0]
        self.parameters = parameters
        self.m = tf.cast(tf.shape(self.a0)[0], tf.float32)
        self.k = tf.cast(k, tf.float32)
        self.regression = regression

        if update_fun is None:
            self.update_parameters = self.update_parameters_default
        else:
            self.update_parameters = update_fun

        self.make_initial_kalman_state()

        self.initial_smoother_state = (
            tf.zeros_like(self.a0), 
            tf.zeros_like(self.P0),
            tf.zeros_like(self.P0)
        )

        self.noise_state_dep = noise_state_dep

    def make_initial_kalman_state(self):

        # initialize Kalman state
        a0 = self.a0
        Ht, Qt, Zt, Tt = self.parameters
        H, Q, Z, T = Ht(a0, 0), Qt(a0, 0), Zt(a0, 0), Tt(a0, 0)

        K = tf.ones([
            tf.shape(self.a0)[0], 
            tf.shape(H)[0]
        ])

        v = tf.ones([
            tf.shape(H)[0],
            1
        ])
        
        t0 = tf.constant(0)
        ll0 = tf.constant([[0.]])

        self.initial_kalman_state = (
            self.a0, self.P0, ll0, t0,
            v, self.a0, self.P0,
            self.a0, self.P0, self.P0
        )

    def make_sigma_points(self, m, at, Pt):
        
        try:
            Pts = tf.linalg.cholesky(Pt)
        except:
            pdb.set_trace()

        x0 = at
        x1 = at + Pts * tf.sqrt(
            tf.cast(m + self.k, tf.float32))
        x2 = at - Pts * tf.sqrt(
            tf.cast(m + self.k, tf.float32))
        X = tf.concat([x0, x1, x2], axis=1)

        w0 = tf.ones([1, 1]) * self.k / (m + self.k)
        w12 = tf.ones([1, 2 * m]) * 1 / (
            2 * (m + self.k))

        W = tf.concat([w0, w12], axis=1)

        return X, W

    def get_uk_filter_step(self, params):
    
        def _uk_filter_step(kalman_state, y):

            '''
            Performs one Kalman filter step
            '''
        
            H, Q, Z, T = params
            
            a_prior, P_prior, ll, t, _, _, _, _, _, _ = kalman_state

            X, W = self.make_sigma_points(self.m, a_prior, P_prior)

            yb = tf.reduce_sum(
                Z(X, t) * W, axis=1, keepdims=True)
            u = Z(X, t) - yb
            v = y - yb

            P_av = tf.matmul(
                (X - a_prior) * W,
                u, transpose_b=True
            )

            if not self.noise_state_dep:
                Hx = H(a_prior, t)
            else:
                Hfun = lambda x : (x[1] * H(x[0][:, None], t), 0.)
                Heval = tf.map_fn(
                    Hfun, (
                        tf.transpose(X),
                        tf.transpose(W)
                    )
                )
                Hx = tf.reduce_sum(Heval[0], axis=0)

            P_vv = tf.matmul(
                u * W, u, transpose_b=True
            ) + Hx

            try:
                P_vv_inv = tf.linalg.inv(P_vv)
            except:
                pdb.set_trace()

            P_vv_inv = tf.linalg.inv(P_vv)

            apo = a_prior + tf.matmul(
                P_av, tf.matmul(
                    P_vv_inv, v
                )
            )

            Ppo = P_prior - tf.matmul(
                P_av, tf.matmul(
                    P_vv_inv, P_av,
                    transpose_b=True
                )
            )

            Ppo = Ppo + 1e-4 * tf.eye(tf.shape(Ppo)[0])

            X, W = self.make_sigma_points(self.m, apo, Ppo)

            apr = tf.reduce_sum(
                T(X, t) * W, axis=1, keepdims=True
            )

            if not self.noise_state_dep:
                Qx = Q(a_prior, t)
            else:
                Qfun = lambda x : (x[1] * Q(x[0][:, None], t), 0.)
                Qeval = tf.map_fn(
                    Qfun, (
                        tf.transpose(X),
                        tf.transpose(W)
                    )
                )
                Qx = tf.reduce_sum(Qeval[0], axis=0)

            Ppr = tf.matmul(
                (T(X, t) - apr) * W, 
                T(X, t) - apr, transpose_b=True
            ) + Qx

            Ppr = Ppr + 1e-4 * tf.eye(tf.shape(Ppr)[0])

            Ct1 = tf.matmul(
                (X - apo) * W,
                T(X, t) - apr, transpose_b=True
            )

            ll = - 0.5 * (
                tf.linalg.logdet(P_vv) + 
                tf.matmul(v, tf.matmul(P_vv_inv, v), transpose_a=True)
            )
            
            t += 1

            return (apr, Ppr, ll, t, v, a_prior, P_prior, apo, Ppo, Ct1)
        
        return _uk_filter_step

    def run_kalman_filter(self, params):
        
        _uk_filter_step = self.get_uk_filter_step(params)
        
        self.filtered = tf.scan(
            _uk_filter_step, self.y_ph, self.initial_kalman_state
        )

        return self.filtered

    def get_kalman_smoother_step(self, params):

        def _kalman_smoother_step(smoother_state, filter_state):

            apr, Ppr, _, t, _, _, _, apo, Ppo, Ct1 = filter_state
            atp1_smooth, Ptp1_smooth, _ = smoother_state

            Ptp1_inv = tf.linalg.inv(Ppr)
            G = tf.matmul(Ct1, Ptp1_inv)
            
            if tf.math.equal(t, self.Nt):
                # deal with last step
                at_smooth = apo
                Pt_smooth = Ppo
            else:
                at_smooth = apo + tf.matmul(
                    G, (atp1_smooth - apr)
                )
                Pt_smooth = Ppo + tf.matmul(
                    G, tf.matmul(
                        Ptp1_smooth - Ppr, G,
                        transpose_b=True
                    )
                )

            return (at_smooth, Pt_smooth, G)

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


    def numerical_gradients(self, params, dx=1e-4):
        
        _params_np = params.numpy()
        grads = []
        for i, p in enumerate(_params_np):
            _params_np_l = deepcopy(_params_np)
            _params_np_h = deepcopy(_params_np)

            _params_np_l[i] = p - dx
            _params_np_h[i] = p + dx
            loss_h = self.loss_eager(
                tf.constant(_params_np_h))
            loss_l = self.loss_eager(
                tf.constant(_params_np_l))

            grad = (loss_h - loss_l) / (2 * dx)
            grads.append(grad)

        return tf.stack(grads)

    def log_prob_eager(self, params, dx=1e-4):
        
        with tf.GradientTape(persistent=True) as tape:
            
            tape.watch(params)
            print(params)
            _params = self.update_parameters(params)

            _ = self.run_kalman_filter(_params)

            loss = - tf.reduce_mean(
                self.filtered[2]
                )
            
            print('loss = %s' % loss.numpy())

        grad = tape.gradient(loss, params)
        if (np.any(np.isnan(grad.numpy())) or 
            np.any(np.abs(grad.numpy()) > 1e4)
            ):  
            grad = self.numerical_gradients(params, dx=dx)
            
            print('Numerically estimating gradients')

        return loss, grad

    def loss_eager(self, params):

        _params = self.update_parameters(params)

        _ = self.run_kalman_filter(_params)

        loss = - tf.reduce_mean(
            self.filtered[2]
            )

        return loss

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

    def expectation_step(self, params_current, pcross=None):

        _params_current = self.update_parameters(params_current)

        smoothed = self.run_kalman_smoother(_params_current)

        # Evaluate the Gaussian integrals with sigma point approximations
        a_smoothed = smoothed[0].numpy()
        P_smoothed = smoothed[1].numpy()
        G = smoothed[2].numpy()
        P_cross = []

        sig_points_transition = []
        for t in range(1, a_smoothed.shape[0]):
            mean_transition = tf.concat([a_smoothed[t, :, :], a_smoothed[t-1, :, :]], axis=0)
            if pcross is None:
                S1 = tf.concat([P_smoothed[t, :, :],
                    tf.matmul(P_smoothed[t, :, :],
                            G[t-1, :, :], transpose_b=True)], axis=1)
                S2 = tf.concat([tf.matmul(G[t-1, :, :],
                                        P_smoothed[t, :, :]),
                                P_smoothed[t-1, :, :]], axis=1)
            else:
                S1 = tf.concat([P_smoothed[t, :, :],
                    pcross[t]], axis=1)
                S2 = tf.concat([pcross[t],
                                P_smoothed[t-1, :, :]], axis=1)
            S_transition = tf.concat([S1, S2], axis=0)
            P_cross.append(tf.matmul(G[t-1, :, :],
                                    P_smoothed[t, :, :]))

            X, W = self.make_sigma_points(2 * self.m, mean_transition, S_transition)
            sig_points_transition.append([X, W])

        sig_points_measurement = []
        for t in range(0, a_smoothed.shape[0]):
            X, W = self.make_sigma_points(self.m, a_smoothed[t, :, :], P_smoothed[t, :, :])
            sig_points_measurement.append([X, W])

        return a_smoothed, P_smoothed, sig_points_measurement, sig_points_transition, G, P_cross

    def maximization_step(self, params_current, params_smooth=None, pcross=None):
        
        if params_smooth is None:
            a_smoothed, P_smoothed, sig_points_measurement, sig_points_transition, G, P_cross = self.expectation_step(params_current, pcross=pcross)
            _params = self.update_parameters(params_current)
        else:
            a_smoothed, P_smoothed, sig_points_measurement, sig_points_transition, G, P_cross = self.expectation_step(params_smooth, pcross=pcross)
            _params = self.update_parameters(params_smooth)

        H, Q, Z, T = _params

        m_dim = tf.cast(self.m, tf.int32)

        Sigma = 0
        Phi = 0
        Theta = 0
        B = 0
        C = 0
        D = 0

        H_star = 0

        for t in range(1, a_smoothed.shape[0]):
            X_m, W_m = sig_points_measurement[t-1]
            X_t, W_t = sig_points_transition[t-1]
            
            Sigma_t = P_smoothed[t, :, :] + tf.matmul(
                a_smoothed[t, :, :], a_smoothed[t, :, :], transpose_b=True) 
            Sigma += Sigma_t

            Phi_t = tf.matmul(
                W_m * T(X_m, t, non_linear_only=True),
                T(X_m, t, non_linear_only=True),
                transpose_b=True
            ) 
            Phi += Phi_t

            Theta_t = tf.matmul(
                W_m * Z(X_m, t, non_linear_only=True),
                Z(X_m, t, non_linear_only=True),
                transpose_b=True
            )
            Theta += Theta_t

            B_t = tf.matmul(
                W_m * self.y_ph[t, :, :],
                Z(X_m, t, non_linear_only=True),
                transpose_b=True
            )
            B += B_t

            C_t = tf.matmul(
                W_t * X_t[0:m_dim, :],
                T(X_t[m_dim:, :], t, non_linear_only=True),
                transpose_b=True
            )
            C += C_t

            D_t = tf.matmul(
                self.y_ph[t, :, :],
                self.y_ph[t, :, :],
                transpose_b=True
            )
            D += D_t

            if self.regression:
                Z_lin = params_current['Z_lin'][t]

                H_star += (
                    D_t - tf.matmul(B_t, Z_lin, transpose_b=True) -
                    tf.matmul(Z_lin, B_t, transpose_b=True) +
                    tf.matmul(Z_lin, tf.matmul(Theta_t, Z_lin, transpose_b=True))
                )

        time_steps = a_smoothed.shape[0]

        Sigma = Sigma / time_steps
        Phi = Phi / time_steps
        Theta = Theta / time_steps
        B = B / time_steps
        C = C / time_steps
        D = D / time_steps

        Z_lin_star_r, T_lin_star_r = self.optimal_ZT(C, B, Phi, Theta)

        Q_star = (
            Sigma - tf.matmul(C, T_lin_star_r, transpose_b=True) -
            tf.matmul(T_lin_star_r, C, transpose_b=True) +
            tf.matmul(T_lin_star_r, tf.matmul(Phi, T_lin_star_r, transpose_b=True))
        )

        if self.regression:
            H_star = H_star / time_steps
        else:
            H_star = (
                D - tf.matmul(B, Z_lin_star_r, transpose_b=True) -
                tf.matmul(Z_lin_star_r, B, transpose_b=True) +
                tf.matmul(Z_lin_star_r, tf.matmul(Theta, Z_lin_star_r, transpose_b=True))
            )

        a_0_star = (
            a_smoothed[0, :, :]
        )
        self.a0 = a_0_star

        P0_star = (
            P_smoothed[0, :, :] + tf.matmul(
                a_smoothed[0, :, :] - self.a0,
                a_smoothed[0, :, :] - self.a0,
                transpose_b=True
            )
        )

        P0_current = deepcopy(self.P0)
        self.P0 = P0_star
        
        params_new = deepcopy(params_current)

        H_star_r, Q_star_r = self.optimal_HQ(H_star, Q_star)
        
        params_new['Q'] = Q_star_r
        params_new['H'] = H_star_r
        if not self.regression:
            params_new['Z_lin'] = Z_lin_star_r
        params_new['T_lin'] = T_lin_star_r

        obj = self.em_objective(time_steps, params_new, H_star, Q_star, P0_star)
        obj_pre = self.em_objective_pre(time_steps, params_current, H_star, Q_star, P0_current)

        self.params_record = params_new

        return obj, obj_pre, params_new

    def em_objective_pre(self, time_steps, params_current, H_star, Q_star, P0_current):
        pass


    def em_objective(self, time_steps, params_new, H_star, Q_star, P0_star):

        # Evaluate the objective function at the new optimal parameters

        ll_intial_conditions = - 0.5 * tf.linalg.logdet(self.P0)
        ll_intial_conditions += - 0.5 * tf.trace(
            tf.matmul(
                tf.linalg.inv(self.P0), P0_star
            )
        )

        ll_measurement = - 0.5 * tf.linalg.logdet(params_new['Q'])
        ll_measurement += - 0.5 * tf.trace(
            tf.matmul(
                tf.linalg.inv(params_new['Q']), Q_star
            )
        )
        ll_measurement = ll_measurement * time_steps

        ll_transition = - 0.5 * tf.linalg.logdet(params_new['H'])
        ll_transition += - 0.5 * tf.trace(
            tf.matmul(
                tf.linalg.inv(params_new['H']), H_star
            )
        )
        ll_transition = ll_transition * time_steps

        obj = -(ll_intial_conditions + ll_measurement + ll_transition)

        return obj

    def optimal_HQ(self, H, Q, args=None):

        # Optimal H, Q if restricted to diagonal.

        Q_diag = tf.diag(tf.linalg.diag_part(Q))
        H_diag = tf.diag(tf.linalg.diag_part(H))

        return H_diag, Q_diag

    def optimal_ZT(self, C, B, Phi, Theta, args=None):

        T_lin_star = tf.matmul(C, tf.linalg.inv(Phi))
        Z_lin_star = tf.matmul(B, tf.linalg.inv(Theta))

        return Z_lin_star, T_lin_star

    def get_objective_EM_numerical(self, params_current):

        a_smoothed, P_smoothed, sig_points_measurement, sig_points_transition, G, P_cross = self.expectation_step(params_current)

        def _objective_EM(params):

            _params = self.update_parameters(params)
            H, Q, Z, T = _params

            ll_intial_conditions = - 0.5 * tf.linalg.logdet(self.P0)
            ll_intial_conditions += - 0.5 * tf.trace(
                tf.matmul(
                    tf.linalg.inv(self.P0),
                    P_smoothed[0, :, :] + tf.matmul(
                        a_smoothed[0, :, :] - self.a0,
                        a_smoothed[0, :, :] - self.a0,
                        transpose_b=True
                    )
                )
            )

            ll_transition = 0
            ll_measurement = 0
            for t in range(1, a_smoothed.shape[0]):
                # Assume Q is not a function of state - evaluate at initial state.
                Qt = Q(self.a0, t)
                ll_transition += - 0.5 * tf.linalg.logdet(Qt)
                X, W = sig_points_transition[t-1]
                m_dim = tf.cast(self.m, tf.int32)
                cov_transition = tf.matmul(
                    (X[0:m_dim, :] - T(X[m_dim:, :], t)) * W,
                    (X[0:m_dim, :] - T(X[m_dim:, :], t)),
                    transpose_b=True
                )
                ll_transition += - 0.5 * tf.trace(
                    tf.matmul(
                        tf.linalg.inv(Qt),
                        cov_transition
                    )
                )

                # Assume H is not a function of state - evaluate at initial state.
                Ht = H(self.a0, t)
                ll_measurement += - 0.5 * tf.linalg.logdet(Ht)
                X, W = sig_points_measurement[t-1]
                cov_measurement = tf.matmul(
                    (self.y_ph[t, :, :] - Z(X, t)) * W,
                    (self.y_ph[t, :, :] - Z(X, t)),
                    transpose_b=True 
                )
                
                ll_transition += - 0.5 * tf.trace(
                    tf.matmul(
                        tf.linalg.inv(Ht),
                        cov_measurement
                    )
                )


            total_objective = - (
                ll_intial_conditions +
                ll_transition +
                ll_measurement
            )

            return total_objective

        return _objective_EM

