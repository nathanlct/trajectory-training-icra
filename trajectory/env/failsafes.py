import numpy as np
import math


def safe_velocity(this_vel, lead_vel, headway, max_decel, time_step, min_gap=2.5):
    """Compute a safe velocity for the vehicles.

    Finds maximum velocity such that if the lead vehicle were to stop
    instantaneously, we can bring the following vehicle to rest at the point at
    which the headway is zero.

    WARNINGS:
    1. We assume the lead vehicle has the same deceleration capabilities as our vehicles
    2. We solve for this value using the discrete time approximation to the dynamics. We assume that the
        integration scheme induces positive error in the position, which leads to a slightly more conservative
        driving behavior than the continuous time approximation would induce. However, the continuous time
        safety rule would not be strictly safe.

    Parameters
    ----------
    env : flow.envs.Env
        current environment, which contains information of the state of the
        network at the current time step

    Returns
    -------
    float
        maximum safe velocity given a maximum deceleration, delay in
        performing the breaking action, and speed limit
    """
    # TODO(eugenevinitsky) hardcoding
    brake_distance = valid_brake_distance(lead_vel, max_decel, time_step)
    v_safe = maximum_safe_stop_speed(headway + brake_distance - min_gap, this_vel, time_step, max_decel)

    return v_safe


def valid_brake_distance(speed, max_deaccel, sim_step):
    """Return the distance needed to come to a full stop if braking as hard as possible. We assume the delay is a time_step.

    Parameters
    ----------
    speed : float
        ego speed
    max_deaccel : float
        maximum deaccel of the vehicle
    delay : float
        the delay before an action is executed
    sim_step : float
        size of simulation step

    Returns
    -------
    float
        the distance required to stop
    """

    # how much we can reduce the speed in each timestep
    speedReduction = max_deaccel * sim_step
    # how many steps to get the speed to zero
    steps_to_zero = int(speed / speedReduction)
    return sim_step * (steps_to_zero * speed - speedReduction * steps_to_zero * (steps_to_zero + 1) / 2) + \
            speed * sim_step


def maximum_safe_stop_speed(brake_distance, speed, sim_step, max_decel):
    """Compute the maximum speed that you can travel at and guarantee no collision.

    Parameters
    ----------
    brake_distance : float
        total distance the vehicle has before it must be at a full stop
    speed : float
        current vehicle speed
    sim_step : float
        simulation step size in seconds

    Returns
    -------
    v_safe : float
        maximum speed that can be travelled at without crashing
    """
    v_safe = maximum_safe_stop_speed_euler(brake_distance, sim_step, max_decel)
    return v_safe


def maximum_safe_stop_speed_euler(brake_distance, sim_step, max_decel):
    """Compute the maximum speed that you can travel at and guarantee no collision for euler integration.

    Parameters
    ----------
    brake_distance : float
        total distance the vehicle has before it must be at a full stop
    sim_step : float
        simulation step size in seconds

    Returns
    -------
    v_safe : float
        maximum speed that can be travelled at without crashing
    """
    if brake_distance <= 0:
        return 0.0

    speed_reduction = max_decel * sim_step

    s = sim_step
    t = sim_step

    # h = the distance that would be covered if it were possible to stop
    # exactly after gap and decelerate with max_deaccel every simulation step
    # h = 0.5 * n * (n-1) * b * s + n * b * t (solve for n)
    # n = ((1.0/2.0) - ((t + (pow(((s*s) + (4.0*((s*((2.0*h/b) - t)) + (t*t)))), (1.0/2.0))*sign/2.0))/s))
    sqrt_quantity = math.sqrt(
    ((s * s) + (4.0 * ((s * (2.0 * brake_distance / speed_reduction - t)) + (t * t))))) * -0.5
    n = math.floor(.5 - ((t + sqrt_quantity) / s))
    h = 0.5 * n * (n - 1) * speed_reduction * s + n * speed_reduction * t
    assert (h <= brake_distance + 1e-6)
    # compute the additional speed that must be used during deceleration to fix
    # the discrepancy between g and h
    r = (brake_distance - h) / (n * s + t)
    x = n * speed_reduction + r
    assert (x >= 0)
    return x
