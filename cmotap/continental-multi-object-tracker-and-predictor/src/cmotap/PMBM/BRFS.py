'''
Copyright (C) 2024 Continental Automotive GmbH.
Licensed under the BSD-3-Clause License.
@author: Reichardt Joerg
'''

import numpy as np


class BRFS(object):
    """A Bernoulli Random Finite Set of cardinality at most 1"""

    def __init__(self, r, state_density, t_birth=0, t_death=0, w_death=1):
        """Initializes a Bernoulli Random Finite Ste

        Parameters
        ----------
        r: float
            probability of existence

        state_density: object
            an objecte implementing the Density interface

        t_birth: int
            cycle in which the object is born

        t_death: int
            cycle in which the object dies

        w_death: float
            probability of death
        """
        self._r = r
        self._state_density = state_density
        self._t_birth = t_birth
        self._t_death = (
            t_death if isinstance(t_death, np.ndarray) else np.array([t_death])
        )
        self._w_death = (
            w_death if isinstance(w_death, np.ndarray) else np.array([w_death])
        )

    def __repr__(self):
        return "r:\n{}\n{}\n{}, {}, {}".format(
            self._r,
            self._state_density,
            self._t_birth,
            self._t_death[-5:],
            self._w_death[-5:],
        )

    @property
    def r(self):
        """Probabiliyt of existence"""
        return self._r

    @property
    def state_density(self):
        """State density of the object in the set"""
        return self._state_density

    def predict(self, motion_model, dt, P_S, r_min=1e-4):
        """Perform prediction step

        Parameters
        ----------

        motion_model: object
            an object implementing the MotionModel interface

        dt: float
            time to predict forward by

        P_S: float
            probability of survival in sensor range

        r_min: float
            minimum probability of existence


        Returns
        -------

            An updated BRFS Object
        """
        if self._w_death[-1] >= r_min:
            return BRFS(
                self.r,
                self.state_density.predict(motion_model, dt=dt),
                self._t_birth,
                np.block([self._t_death, self._t_death[-1] + 1]),
                np.block(
                    [
                        self._w_death[:-1],
                        self._w_death[-1] * (1 - P_S),
                        self._w_death[-1] * P_S,
                    ]
                ),
            )
        else:
            return BRFS(
                self._r,
                self._state_density,
                self._t_birth,
                self._t_death,
                self._w_death,
            )

    def compensate_sensor_movement(self, sensor_movement):
        """Compensate 2D sensor Movement

        this function calles compensate_sensor_movement on the state density
        object within the BRFS, see the documentation there
        """
        return BRFS(
            self.r,
            self.state_density.compensate_sensor_movement(sensor_movement),
            self._t_birth,
            self._t_death,
            self._w_death,
        )

    def undetected_update(self, sensor_model):
        """Update BRFS for an undetected object

        Parameters
        ----------
        sensor_model: object
            An object implementing the SensorModel interface

        Returns
        -------
        tuple of updated BRFS and log Likelihood of being undetected, i.e.
        being either non-existent or misdetected
        """
        lik_nodetect = self.r * (1 - sensor_model.P_D * self._w_death[-1])
        lik_undetected = 1 - self.r + lik_nodetect
        r_new = lik_nodetect / lik_undetected
        return BRFS(
            r_new,
            self.state_density,
            self._t_birth,
            self._t_death,
            np.block([self._w_death[:-1], self._w_death[-1] * (1 - sensor_model.P_D)])
            / (1 - self._w_death[-1] * sensor_model.P_D),
        ), np.log(lik_undetected)

    def detected_update_lik(self, z, observation_model, sensor_model):
        """Loglikelihood of observation

        Parameters
        ----------

        z: nparray
            observation vector

        observation_model: object
            an object implementing the ObservationModel interface

        sensor_model: object
            an object implementing the SensorModel interface
        """
        return self.state_density.observationLogLikelihood(
            z, observation_model
        ) + np.log(sensor_model.P_D * self.r * self._w_death[-1])

    def detected_update_state(self, z, observation_model):
        """Update BRFS for a detected object

        Parameters
        ----------

        z: ndarray
            observation vector

        observation_model: object
            an object implementing the ObservationModel interface

        Returns
        -------

            an updated BRFS
        """
        return BRFS(
            1.0,
            self.state_density.update(z, observation_model),
            self._t_birth,
            self._t_death[-1],
            1,
        )

    def draw(self, ax, **kwargs):
        """Draws the BRFS"""
        self.state_density.draw(ax=ax, **kwargs)
