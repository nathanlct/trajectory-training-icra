import numpy as np
from trajectory.env.failsafes import safe_velocity


class IDMController(object):
    def __init__(self, v0=45, T=1, a=1.3, b=2.0, delta=4, s0=2, noise=0.3):
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.noise = noise
        self.accel_without_noise = 0

    def get_accel(self, this_vel, lead_vel, headway, sim_step):
        """See parent class."""
        # in order to deal with ZeroDivisionError
        if abs(headway) < 1e-3:
            headway = 1e-3

        if lead_vel is None:  # no car ahead
            s_star = 0
        else:
            s_star = self.s0 + max(
                0, this_vel * self.T + this_vel * (this_vel - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        accel = self.a * (1 - (this_vel / self.v0)**self.delta - (s_star / headway)**2)

        self.accel_without_noise = accel

        if self.noise > 0:
            accel += np.sqrt(sim_step) * np.random.normal(0, self.noise)
        return accel

    def get_accel_without_noise(self):
        """
        Return the accel without applying any noise.
        Must be called after get_accel to updated result.
        """
        return self.accel_without_noise


class TimeHeadwayFollowerStopper(object):
    """New FollowerStopper with safety envelopes based on time-headways.

    Usage
    -----
    See base class for example.

    Parameters
    ----------
    veh_id : str
        unique vehicle identifier
    v_des : float, optional
        desired speed of the vehicles (m/s)
    no_control_edges : [str]
        list of edges that we should not apply control on
    """

    def __init__(self,
                 v_des=15,
                 max_accel=1.5,
                 max_deaccel=3.0):

        # other parameters
        self.h_1 = 0.4
        self.h_2 = 0.6
        self.h_3 = 0.8
        self.dx_1_0 = 4.5
        self.dx_2_0 = 5.25
        self.dx_3_0 = 6.0
        self.d_1 = 1.5
        self.d_2 = 1.0
        self.d_3 = 0.5

        self.v_des = v_des
        self.max_accel = max_accel
        self.max_deaccel = max_deaccel

        self.accel = 0

    def get_accel(self, this_vel, lead_vel, headway, time_step):
        """See parent class."""

        dx = headway
        dv_minus = min(lead_vel - this_vel, 0)

        dx_1 = 1 / (2 * self.d_1) * dv_minus ** 2 + max(self.dx_1_0, self.h_1 * this_vel)
        dx_2 = 1 / (2 * self.d_2) * dv_minus ** 2 + max(self.dx_2_0, self.h_2 * this_vel)
        dx_3 = 1 / (2 * self.d_3) * dv_minus ** 2 + max(self.dx_3_0, self.h_3 * this_vel)
        v = min(max(lead_vel, 0), self.v_des)
        # compute the desired velocity
        if dx <= dx_1:
          v_cmd = 0
        elif dx <= dx_2:
          v_cmd = v * (dx - dx_1) / (dx_2 - dx_1)
        elif dx <= dx_3:
          v_cmd = v + (self.v_des - v) * (dx - dx_2) \
                  / (dx_3 - dx_2)
        else:
          v_cmd = self.v_des

        v_safe = safe_velocity(this_vel, lead_vel, headway, self.max_deaccel, time_step)
        desired_accel = np.clip((v_cmd - this_vel) / time_step, -np.abs(self.max_deaccel), self.max_accel)
        v_next = desired_accel * time_step + this_vel
        if v_next > v_safe:
          self.accel = np.clip((v_safe - this_vel) / time_step, -np.abs(self.max_deaccel), self.max_accel)
        else:
          self.accel = np.clip((v_cmd - this_vel) / time_step, -np.abs(self.max_deaccel), self.max_accel)

        return self.accel

    def get_accel_without_noise(self):
        """
        Return the accel without applying any noise.
        Must be called after get_accel to updated result.
        """
        return self.accel
