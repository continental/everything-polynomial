'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''

import numpy as np

from src.cmotap import utils
from src.cmotap.PMBM.BRFS import BRFS


class PPP(object):
    """A Poisson Point Process"""

    def __init__(self, log_w, state_densities):
        """Initialize a Poisson Point Process (PPP)

        Parameters
        ----------
        log_w: list
            component log probabilities

        state_densities: list of objects implementing Density interface
        """
        self._log_w = np.array(log_w)
        self._state_densities = state_densities

    def __repr__(self):
        return "\n".join(
            [
                "log_w:{}\n{}".format(w_i, c_i)
                for (w_i, c_i) in zip(self._log_w, self._state_densities)
            ]
        )

    @property
    def log_w(self):
        """Mixture Component log probabilies"""
        return self._log_w

    @property
    def state_densities(self):
        """Mixture Components"""
        return self._state_densities

    def predict(self, motion_model, dt, birth_model, P_S):
        """Predict PPP components forward in time

        Paramters
        ---------
        motion_model: object
            an object implementing the MotionModel interface

        dt: float
            time delta to predict forward

        birth_model: object
            an object implementing the BirthModel interface

        P_S: float
            probability of object surival in sensor range

        Returns
        -------
            an updated PPP object
        """
        log_w = np.block([self._log_w + np.log(P_S), birth_model.log_w])
        state_densities = [
            c_i.predict(motion_model, dt) for c_i in self.state_densities
        ] + birth_model.state_densities
        return PPP(log_w, state_densities)

    def undetected_update(self, sensor_model):
        """Update undetected objects

        Parameters
        ----------
        sensor_model: object
            an object implementing the SensorModel interface

        Returns
        -------
        an updated PPP object
        """
        return PPP(self.log_w + np.log(1 - sensor_model.P_D), self.state_densities)

    def detected_update(self, k, indices, z, observation_model, sensor_model):
        """Generate a new object from a detection

        Parameters
        ----------
        k: int
            update cycle

        indices: vector
            boolean vector indicating which components to use for generating new objects

        z: ndarray
            observation vector

        observation_model: object
            an object implementing the ObservationModel interface

        sensor_model: object
            an object implementing the SensorModel interface

        Returns
        -------
        a BRFS object with the new object
        the likelihood of the new object
        """
        assert np.sum(indices) > 0, "You need detections for this!"
        assert len(indices) == len(self.state_densities)
        states_upd = [
            c_i.update(z, observation_model)
            for (c_i, use_component) in zip(self.state_densities, indices)
            if use_component
        ]
        log_w_upd = [
            c_i.observationLogLikelihood(z, observation_model)
            + log_w_i
            + np.log(sensor_model.P_D)
            for (c_i, log_w_i, use_component) in zip(
                self.state_densities, self.log_w, indices
            )
            if use_component
        ]

        log_w_upd, log_sum_w_upd = utils.normalize_logweights(log_w_upd)
        _, lik_new = utils.normalize_logweights(
            np.block([log_sum_w_upd, np.log(sensor_model.intensity_c)])
        )

        r_new = np.exp(log_sum_w_upd - lik_new)
        state_density_new = self._state_densities[0].__class__.from_mixture_density(
            log_w_upd, states_upd
        )
        return BRFS(r_new, state_density_new, k, k, 1), lik_new

    def ellipsoidal_gating(self, z, observation_model, gating_size):
        """Check which observations can be considered for updates

        Parameters
        ----------
        z: ndarray
            observation matrix

        observation_model: object
            an object implementing the ObservationModel interface

        gating_size: float
            size of gate for measruements to be considered for updates
        """
        m = z.shape[1]
        gating_matrix = np.zeros((m, len(self.state_densities))).astype(bool)
        for i, c in enumerate(self.state_densities):
            _, gating_matrix[:, i] = c.ellipsoidal_gating(
                z, observation_model, gating_size
            )
        meas_in_gate = np.any(gating_matrix, axis=1)

        return gating_matrix, meas_in_gate

    def calculate_association_cost_matrix(
        self, z, k, gating_matrix, observation_model, sensor_model
    ):
        """
        Parameters
        ----------

        z: ndarray
            observation matrix

        k: int
            cycle time

        gating_matrix: ndarray
            output of ellipsoidal gating

        observation_model: object
            an object implementing the ObservationModel interface

        sensor_model: object
            an object implementing the SensorModel interface

        Returns
        -------
        L: ndarray
            cost matrix

        new_detection_hypotheses: list
            list of BRFS for potential new objects
        """
        m = z.shape[1]
        L_new = np.ones((m, m)) * np.inf
        meas_in_gate = np.any(gating_matrix, axis=1)
        new_detection_hypotheses = []
        for i, z_i in enumerate(z.T):
            if meas_in_gate[i]:
                new_hyp, log_likelihood = self.detected_update(
                    k, gating_matrix[i, :], z_i, observation_model, sensor_model
                )
            else:
                new_hyp = BRFS(
                    -1, self.state_densities[0].__class__(), k, k, -1
                )  # dummy
                log_likelihood = np.log(sensor_model.intensity_c)

            new_detection_hypotheses.append([new_hyp])
            L_new[i, i] = -log_likelihood
        return L_new, new_detection_hypotheses

    def prune(self, threshold=np.log(1e-3)):
        """Discard components of low prabability

        Parameters
        ----------
        threshold: float
            minimum log probability of components to be kept
        """
        self._log_w, self._state_densities = utils.prune_multi_hypotheses(
            self._log_w, self._state_densities, threshold, normalize_log_w=False
        )

    def cap(self, M=100):
        """Limit the number of components

        Parameters
        ----------
        M: int
            maximum number of components to keep
        """
        self._log_w, self._state_densities = utils.cap_multi_hypotheses(
            self._log_w, self._state_densities, M, normalize_log_w=False
        )

    def merge(self, threshold=2):
        """Merge similar components

        Parameters
        ----------
        threshold: float
            similarity threshold for components to be considered for merge
        """
        self._log_w, self._state_densities = utils.merge_multi_hypotheses(
            self._log_w, self._state_densities, threshold
        )

    def draw(self, ax):
        """Draw the PPP density"""
        for s in self.state_densities:
            s.draw(ax=ax, color="g")
