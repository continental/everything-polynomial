'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''

import numpy as np

from src.cmotap.dataassociation import dataassociation
from src.cmotap import utils
from src.cmotap.PMBM.MBM import MBM
from src.cmotap.PMBM.PPP import PPP


class PMBM(object):
    """A Poisson-Multi-Bernoulli Object"""

    def __init__(self, ppp, mbm):
        """Initialize an MBM Object

        Parameters
        ----------
            ppp: an object of type PPP (Poisson-Point-Process) for undetected objects
            mbm: an object of type MBM (Multi-Bernoulli-Mixture) for detected objects
        """
        self._undetected = ppp
        self._detected = mbm

    def __repr__(self):
        return "PPP:\n{}\n\nMBM:\n{}".format(self._undetected, self._detected)

    @property
    def undetected(self):
        """the undetected objects (PPP)"""
        return self._undetected

    @property
    def detected(self):
        """the detected objects (MBM)"""
        return self._detected

    def predict(self, motion_model, dt, birth_model, P_S):
        """Predict PMBM forward in time

        Parameters
        ----------
        motion_model: object
            an object that implements the MotionModel interface

        dt: float
            prediction horizon

        birth_model: object
            and object that implements the BirthModel interface

        P_S: float
            probability of object survival in sensor range
        """
        ppp_new = self._undetected.predict(motion_model, dt, birth_model, P_S)
        mbm_new = self._detected.predict(motion_model, dt, P_S)
        return PMBM(ppp_new, mbm_new)

    def compensate_sensor_movement(self, sensor_movement):
        """Compensate the detected object for the sensor movement, undetected objects
        should be in sensor coordinates anyhow

        see the details for compensate_sensor_movement of the state densities
        for details

        Returns
        -------
            A compensated PMBM object
        """
        mbm_new = self._detected.compensate_sensor_movement(sensor_movement)
        return PMBM(self._undetected, mbm_new)

    def update(self, z, k, observation_model, sensor_model, gating_size):
        """Update PMBM with observations

        Parameters
        ----------
        z: ndarray
            observation matrix, observation dim x number of observations

        observation_model: object
            an object implementing the ObservationModel interface

        sensor_model: object
            an object implementing the SensorModel interface

        gating_size: float
            size of gate for dataassociation

        Returns
        -------
            an updated PMBM object
        """

        (
            gating_matrix_undetected,
            meas_in_undetected,
        ) = self.undetected.ellipsoidal_gating(z, observation_model, gating_size)

        gating_matrices_detected, meas_in_detected = self.detected.ellipsoidal_gating(
            z, observation_model, gating_size
        )

        (
            new_local_log_likelihoods,
            new_local_hypotheses,
            offset,
        ) = self.detected.extend_local_hypotheses(
            z, observation_model, sensor_model, gating_matrices_detected
        )

        assert len(new_local_hypotheses) == len(self.detected.local_hypotheses)

        (
            L_new,
            new_detection_hypotheses,
        ) = self.undetected.calculate_association_cost_matrix(
            z, k, gating_matrix_undetected, observation_model, sensor_model
        )

        new_local_hypotheses += new_detection_hypotheses

        # claculate new global hypotheses
        new_log_w = []
        new_global_hypotheses = []

        for global_hypothesis, log_w in zip(
            self.detected.global_hypotheses, self.detected.log_w
        ):

            L_j, lik_tmp = self.detected.calculate_association_cost_matrix(
                z,
                global_hypothesis,
                gating_matrices_detected,
                new_local_log_likelihoods,
            )

            L = np.block([L_j, L_new])

            col4rowAll, row4colAll, gainAll = dataassociation.assign2D(L)
            # col4rowAll, row4colAll, gainAll= kBestAssign2D(L, int(np.ceil(np.exp(log_w) * 100)))
            # col4rowAll, row4colAll, gainAll= kBestAssign2D(L, 1)

            for g, row4col in zip(gainAll, row4colAll.T):
                new_hypothesis = self.detected.make_new_hypothesis(
                    global_hypothesis, row4col
                )

                assert len(new_hypothesis) == L.shape[1]
                new_global_hypotheses.append(new_hypothesis)
                new_log_w.append(log_w - g + lik_tmp)

        if len(self.detected.global_hypotheses) < 1:
            new_log_w = np.array([0])
            new_global_hypotheses = np.zeros((1, len(new_local_hypotheses)))

        new_log_w, _ = utils.normalize_logweights(new_log_w)

        undetected = self.undetected.undetected_update(sensor_model)

        detected = MBM(new_log_w, new_global_hypotheses, new_local_hypotheses)

        # housekeeping
        detected.prune(threshold=np.log(1e-3), normalize_log_w=False)
        detected.cap(M=100, normalize_log_w=True)
        detected.remove_untracked_targets()
        detected.remove_unused_local_hypotheses()

        return PMBM(undetected, detected)

    def estimate(self, threshold=0.4):
        """Estimate of most likely global hypothesis for detected objects

        Parameters
        ----------
        threshold: float
            minimum existence probability for objects to be considered

        Returns
        -------
        list of detected object densities
        """
        return self.detected.estimate(threshold=threshold)

    def reduce_and_recycle(self, r_min=1e-3):
        """Housekeeping after each iteration

        Parameters
        ----------
        r_min: float
            discard all objects with existence probability < r_min

        Returns
        -------
        a reduced PMBM object
        """
        # recycle
        recycled_log_w, recycled_state_densities = self.detected.recycle(
            prune_threshold=r_min, recycle_threshold=0.4
        )

        undetected = PPP(
            np.block([self.undetected.log_w, recycled_log_w]),
            self.undetected.state_densities + recycled_state_densities,
        )
        undetected.prune(np.log(1e-3))
        undetected.merge(2)

        return PMBM(undetected, self.detected)
