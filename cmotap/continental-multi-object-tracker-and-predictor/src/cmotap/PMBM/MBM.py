'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''

import numpy as np
import matplotlib.pyplot as plt

from src.cmotap import utils


class MBM(object):
    """A Multi-Bernoulli-Mixture"""

    def __init__(self, log_w=[], global_hypotheses=[], local_hypotheses=[]):
        """Initializes an Multi-Bernoulli-Mixture

        Parameters
        ----------
        log_w: list of float
            log probabilities of global hypotheses

        global_hypotheses: list of list of int
            each list of int corresponds to a global hypothesis,
            the indices in a global hypothesis correspond to local hypotheses

        local_hypotheses: list of density
            densities should implement the Density interface
        """
        self._log_w = np.array(log_w)
        self._global_hypotheses = np.array(global_hypotheses, dtype=np.int64)
        self._local_hypotheses = local_hypotheses
        self._cmap = plt.get_cmap("tab20")

    @property
    def log_w(self):
        """Probabilities of global hypotheses"""
        return self._log_w

    @property
    def global_hypotheses(self):
        """Global hypotheses, i.e. lists of local hypotheses"""
        return self._global_hypotheses

    @property
    def local_hypotheses(self):
        """Local hypotheses"""
        return self._local_hypotheses

    def predict(self, motion_model, dt, P_S):
        """Predict local hypotheses forward in time

        Paramters
        ---------
        motion_model: object
            an object implementing the MotionModel interface

        dt: float
            time delta to predict forward

        P_S: float
            probability of object surival in sensor range

        Returns
        -------
            an updated MBM object
        """
        new_local_hypotheses = [
            [hyp.predict(motion_model, dt, P_S) for hyp in target]
            for target in self.local_hypotheses
        ]
        return MBM(self.log_w, self.global_hypotheses, new_local_hypotheses)

    def compensate_sensor_movement(self, sensor_movement):
        """Compensates for 2D sensor movement
        see the corresponding function in the state density for details

        Returns
        -------
            an MBM object adapted to the new sensor position
        """
        new_local_hypotheses = [
            [hyp.compensate_sensor_movement(sensor_movement) for hyp in target]
            for target in self.local_hypotheses
        ]
        return MBM(self.log_w, self.global_hypotheses, new_local_hypotheses)

    def estimate(self, threshold=0.4):
        """Estimate object densities for most likely global hypotheses

        Parameters
        ----------

        threshold: float
            probability of existence for an object to be considered

        Returns
        -------
            list of tuples of object state density & t_death
        """
        best_global_hypothesis_index = np.argmax(self.log_w)
        best_global_hypothesis = self.global_hypotheses[best_global_hypothesis_index]
        estimates = []
        for target_tree_index, hyp_idx in enumerate(best_global_hypothesis):
            if hyp_idx < 0:
                continue
            hyp = self.local_hypotheses[target_tree_index][hyp_idx]
            if hyp.r >= threshold:
                idx_death = np.argmax(hyp._w_death)
                # t_len = hyp._t_death[idx_death] - hyp._t_birth + 1
                estimates.append(
                    (hyp.state_density, hyp._t_death[idx_death])
                    # StateDensity(hyp.state_density._x[:t_len], hyp.state_density._P[:t_len])
                )
        return estimates

    def ellipsoidal_gating(self, z, observation_model, gating_size):
        """
        Parameters
        ----------
        z: ndarray
            multi-object observations, number of observations x observation size

        observation_model: object
            an object implementing the ObservationModel interface

        gating_size:
            size of gate to consider

        Returns
        -------
            list: gating_matrices, one for each local hypothesis
            vector of bool: measurement_in_gate, one entry for each measurement
        """

        m = z.shape[1]
        gating_matrices = []
        meas_in_gate = np.zeros(m).astype(bool)
        for target_tree in self.local_hypotheses:
            gm = np.zeros((m, len(target_tree))).astype(bool)
            for hyp_idx, hyp in enumerate(target_tree):
                _, gm[:, hyp_idx] = hyp.state_density.ellipsoidal_gating(
                    z, observation_model, gating_size
                )
            gating_matrices.append(gm)
            meas_in_gate = meas_in_gate | np.any(gm, axis=1)
        return gating_matrices, meas_in_gate

    def calculate_association_cost_matrix(
        self, z, global_hypothesis, gating_matrices, new_local_log_likelihoods
    ):
        """
        Parameters
        ----------

        z: ndarray
            observation matrix

        global_hypothesis: list
            single global hypothesis, i.e. a list of indices into local hypotheses

        gating_matrices: list of ndarray
            output of ellipsoidal gating

        new_local_log_likelihoods: list of float
            log likelihoods of local hypothese if they were updated with the measurents

        Returns
        -------

        tuple(ndarray, float)
            cost matrix L, undetected likelihood
        """
        m = z.shape[1]
        L_j = np.ones((m, len(global_hypothesis))) * np.inf
        lik_tmp = 0.0
        for target_tree_index, hyp_idx in enumerate(global_hypothesis):
            if hyp_idx < 0:
                continue
            offset = hyp_idx * (m + 1)
            undetected_likelihood = new_local_log_likelihoods[target_tree_index][offset]
            measurement_likelihoods = new_local_log_likelihoods[target_tree_index][
                offset + 1: offset + m + 1
            ]

            measurements_in_gate = gating_matrices[target_tree_index][:, hyp_idx]

            L_j[measurements_in_gate, target_tree_index] = -(
                measurement_likelihoods[measurements_in_gate] - undetected_likelihood
            )
            lik_tmp += undetected_likelihood
        return L_j, lik_tmp

    def extend_local_hypotheses(
        self, z, observation_model, sensor_model, gating_matrices
    ):
        """Update local hypotheses with all potential detections
        Parameters
        ----------
        z: ndarray
            observation matrix

        observation_model: object
            an object that implements the ObservationModel interface

        sensor_model: object
            an object that implements the SensorModel interface

        gating_matrices: ndarray
            output of ellipsoidal_gating

        Returns
        -------
            new_local_loglikelihoods: list of float
            new_local_hypotheses: list of densities
            offset: list of index offsets
        """
        m = z.shape[1]

        offset = []
        new_local_hypotheses = []
        new_local_log_likelihoods = []
        for target_tree, gm in zip(self.local_hypotheses, gating_matrices):

            new_target_tree = [None] * (1 + m) * len(target_tree)
            new_target_likelihoods = np.array(
                [None] * (1 + m) * len(target_tree), dtype=np.float64
            )

            for hyp_idx, hyp in enumerate(target_tree):
                offset = hyp_idx * (m + 1)
                (
                    new_target_tree[offset],
                    new_target_likelihoods[offset],
                ) = hyp.undetected_update(sensor_model)
                offset += 1
                for measurement_index in np.where(gm[:, hyp_idx])[0]:
                    new_target_tree[
                        offset + measurement_index
                    ] = hyp.detected_update_state(
                        z[:, measurement_index], observation_model
                    )
                    new_target_likelihoods[
                        offset + measurement_index
                    ] = hyp.detected_update_lik(
                        z[:, measurement_index], observation_model, sensor_model
                    )

            new_local_hypotheses.append(new_target_tree)
            new_local_log_likelihoods.append(new_target_likelihoods)

        return new_local_log_likelihoods, new_local_hypotheses, offset

    def make_new_hypothesis(self, global_hypothesis, row4col):
        """Form new global hypotheis after data association

        Parameters
        ----------
        global_hypothesis: list
            a single global hypotheis

        row4col: ndarray
            output of dataassociation step

        Returns
        -------
            A single new global hypotheis
        """
        m = row4col.size
        new_hypothesis = []
        for target_tree_index, measurement_index in enumerate(row4col):
            if target_tree_index < len(global_hypothesis):
                hyp_idx = global_hypothesis[target_tree_index]
                offset = hyp_idx * (m + 1)
                if measurement_index < 0:
                    # print('Tree: {}, hyp_idx: {} -> undetected'.format(
                    #   target_tree_index, hyp_idx)
                    # )
                    undetected_idx = offset
                    new_hypothesis.append(undetected_idx)
                else:
                    # print('Tree: {}, hyp_idx: {} -> updated with measurement {}'.format(
                    #   target_tree_index, hyp_idx, measurement_index)
                    # )
                    detected_idx = offset + 1 + measurement_index
                    new_hypothesis.append(detected_idx)
            else:
                if measurement_index < 0:
                    new_hypothesis.append(-1)
                else:
                    new_hypothesis.append(0)
        return new_hypothesis

    def prune(self, threshold=np.log(1e-3), normalize_log_w=True):
        """Prune hypotheses of low probability

        Parameters
        ----------
        threshold: float
            minimum log probability of a hypothesis to be kept

        normalize_log_w: bool
            whether to normalize log probabilities after pruning
        """
        self._log_w, gh = utils.prune_multi_hypotheses(
            self._log_w, self._global_hypotheses, threshold, normalize_log_w
        )
        self._global_hypotheses = np.array(gh, dtype=np.int64)

    def cap(self, M=100, normalize_log_w=True):
        """Limit the number of global hypotheses

        Parameters
        ----------
        M: int
            maximum number of global hypotheses to keep

        normalize_log_w: bool
            whether to renormalize the log probabilities of global hypotheses
        """
        self._log_w, gh = utils.cap_multi_hypotheses(
            self._log_w, self._global_hypotheses, M, normalize_log_w
        )
        self._global_hypotheses = np.array(gh, dtype=np.int64)

    def remove_untracked_targets(self):
        """Remove untracked targets"""
        tracked_targets = np.any(self._global_hypotheses >= 0, axis=0)
        self._global_hypotheses = self._global_hypotheses[:, tracked_targets]
        self._local_hypotheses = [
            local_hypothesis
            for (local_hypothesis, is_tracked) in zip(
                self._local_hypotheses, tracked_targets
            )
            if is_tracked
        ]

    def remove_unused_local_hypotheses(self):
        """Remove unused local hypotheses"""
        for tree_index, local_hypotheses_indices in enumerate(
            self._global_hypotheses.T
        ):
            used_hypothesis = local_hypotheses_indices >= 0
            used_hypothesis_indices = local_hypotheses_indices[used_hypothesis]
            old_indices = np.unique(used_hypothesis_indices)
            index_map = {
                old: new
                for old, new in zip(
                    sorted(old_indices), np.arange(len(old_indices), dtype=np.int64)
                )
            }

            self._global_hypotheses[used_hypothesis, tree_index] = [
                index_map[old_index] for old_index in used_hypothesis_indices
            ]
            self._local_hypotheses[tree_index] = [
                hyp
                for idx, hyp in enumerate(self._local_hypotheses[tree_index])
                if idx in index_map
            ]

    def merge_duplicate_global_hypotheses(self):
        """Merge duplicate global hypotheses"""
        self._global_hypotheses, unique_idx, inverse_idx = np.unique(
            self._global_hypotheses, return_index=True, return_inverse=True, axis=0
        )
        self._log_w = np.array(
            [
                utils.normalize_logweights(self._log_w[inverse_idx == i])[1]
                for i in unique_idx
            ]
        )

    def recycle(self, prune_threshold=1e-3, recycle_threshold=0.4):
        """Recycle dead and dying objects (into undetected objects)

        Parameters
        ----------

        prune_threshold: float
            all local hypotheses with existence probability < prune_threhold
            are discarded

        recycle_threshold: float
            all local hypotheses with existence threshold < recycle_threshold
            are recycled

        Returns
        -------
            tuple(vector, list) of log probabilities and local hypotheses to be recycled
        """
        recycled_log_w = []
        recycled_state_densities = []

        for target_tree_index, local_hypotheses in enumerate(self._local_hypotheses):
            # recycle

            to_recycle = [
                (hyp_index, hyp)
                for (hyp_index, hyp) in enumerate(local_hypotheses)
                if (hyp.r < recycle_threshold and hyp.r >= prune_threshold)
            ]
            for hyp_index, hyp in to_recycle:
                idx = np.where(
                    self._global_hypotheses[:, target_tree_index] == hyp_index
                )[0]
                _, log_w_tmp = utils.normalize_logweights(self._log_w[idx])
                recycled_log_w.append(np.log(hyp.r) + log_w_tmp)
                recycled_state_densities.append(hyp.state_density)

            # prune all things to prune and those recycled
            index_map = [
                -1 if hyp.r < recycle_threshold else hyp_index
                for hyp_index, hyp in enumerate(local_hypotheses)
            ]
            self._global_hypotheses[:, target_tree_index] = [
                -1 if hyp_idx < 0 else index_map[hyp_idx]
                for hyp_idx in self._global_hypotheses[:, target_tree_index]
            ]
            # removal of hypothesis from self._local_hypotheses happens with removal of unused leaves below

        self.remove_untracked_targets()
        self.remove_unused_local_hypotheses()
        self.merge_duplicate_global_hypotheses()

        return np.array(recycled_log_w), recycled_state_densities

    def draw(self, ax, t_limit=-1):
        """Draw current best global hypothesis into axis ax

        Parameters
        ----------
        ax: axis object

        t_limit: int
            limit drawing to objects that died at or after t_limit
        """
        for idx, (s, t_death) in enumerate(self.estimate(0.4)):
            if t_death >= t_limit:
                s.draw(ax=ax, color=self._cmap(idx))
