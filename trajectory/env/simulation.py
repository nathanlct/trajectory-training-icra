from collections import defaultdict
from trajectory.env.vehicles import FSVehicle, FSWrappedRLVehicle, IDMVehicle, RLVehicle, TrajectoryVehicle
from trajectory.env.energy_models import PFM2019RAV4
from trajectory.env.utils import get_last_or


class Simulation(object):
    def __init__(self, timestep):
        """Simulation object

        timestep: dt in seconds
        trajectory: ITERATOR yielding triples (position, speed, accel)
            that will be used for the first vehicle in the platoon (not spawned if this is None)
        """
        self.timestep = timestep
        # vehicles in order, from first in the platoon to last
        self.vehicles = []
        self.vlength = 5

        self.step_counter = 0
        self.time_counter = 0

        self.energy_model = PFM2019RAV4()

        self.data_by_time = []
        self.data_by_vehicle = defaultdict(lambda: defaultdict(list))

        self.vids = 0

    def get_vehicles(self, controller=None):
        if controller is None:
            return self.vehicles
        else:
            return list(filter(lambda veh: veh.controller == controller, self.vehicles))

    def add_vehicle(self, controller='idm', kind=None, tags=None, gap=20, **controller_kwargs):
        """Add a vehicle behind the platoon.

        controller: 'idm' or 'rl' or 'trajectory' (do not use trajectory)
        gap: spawn the vehicle that many meters behind last vehicle in platoon
        controller_kwargs: kwargs that will be passed along to the controller constructor
        """
        vehicle_class = {
            'idm': IDMVehicle,
            'fs': FSVehicle,
            'trajectory': TrajectoryVehicle,
            'rl': RLVehicle,
            'rl_fs': FSWrappedRLVehicle,
            'fs': FSVehicle
        }[controller]

        veh = vehicle_class(
            vid=self.vids,
            controller=controller,
            kind=kind,
            tags=tags,
            pos=0 if len(self.vehicles) == 0 else self.vehicles[-1].pos - gap - self.vlength,
            speed=0 if len(self.vehicles) == 0 else self.vehicles[-1].speed,
            accel=0,
            timestep=self.timestep,
            length=self.vlength,
            leader=None if len(self.vehicles) == 0 else self.vehicles[-1],
            **controller_kwargs)
        if len(self.vehicles) > 0:
            self.vehicles[-1].follower = veh
        self.vids += 1

        self.vehicles.append(veh)
        return veh

    def run(self, num_steps=None):
        running = True
        i = 0
        while running:
            running = self.step()
            i += 1
            if num_steps is not None and i >= num_steps:
                running = False

    def step(self):
        self.step_counter += 1
        self.time_counter += self.timestep

        for veh in self.vehicles[::-1]:
            # update vehicles in reverse order assuming the controller is
            # independant of the vehicle behind you. if at some point it is,
            # then new position/speed/accel have to be calculated for every
            # vehicle before applying the changes
            status = veh.step()
            if status is False:
                return False

        self.collect_data()

        return True

    def add_data(self, veh, key, value):
        self.data_by_vehicle[veh.name][key].append(value)
        # TODO(nl) add data by time as well

    def get_data(self, veh, key):
        return self.data_by_vehicle[veh.name][key]

    def collect_data(self, vehicles=None):
        if vehicles is None:
            vehicles = self.vehicles
        for veh in vehicles:
            self.add_data(veh, 'time', round(self.time_counter, 4))
            self.add_data(veh, 'step', self.step_counter)
            self.add_data(veh, 'id', veh.name)
            self.add_data(veh, 'position', veh.pos)
            self.add_data(veh, 'speed', veh.speed)
            self.add_data(veh, 'accel', veh.accel)
            self.add_data(veh, 'headway', veh.get_headway())
            self.add_data(veh, 'leader_speed', veh.get_leader_speed())
            self.add_data(veh, 'speed_difference', None if veh.leader is None else veh.leader.speed - veh.speed)
            self.add_data(veh, 'leader_id', None if veh.leader is None else veh.leader.name)
            self.add_data(veh, 'follower_id', None if veh.follower is None else veh.follower.name)
            self.add_data(veh, 'instant_energy_consumption', self.energy_model.get_instantaneous_fuel_consumption(veh.accel, veh.speed, 0))
            self.add_data(veh, 'total_energy_consumption', get_last_or(self.data_by_vehicle[veh.name]['total_energy_consumption'], 0) + self.data_by_vehicle[veh.name]['instant_energy_consumption'][-1])
            self.add_data(veh, 'total_distance_traveled', veh.pos - self.data_by_vehicle[veh.name]['position'][0])
            self.add_data(veh, 'total_miles', self.data_by_vehicle[veh.name]['total_distance_traveled'][-1] / 1609.34)
            self.add_data(veh, 'total_gallons', self.data_by_vehicle[veh.name]['total_energy_consumption'][-1] / 3600.0 * self.timestep)
            self.add_data(veh, 'avg_mpg', self.data_by_vehicle[veh.name]['total_miles'][-1] / (self.data_by_vehicle[veh.name]['total_gallons'][-1] + 1e-6))
            self.add_data(veh, 'realized_accel', (veh.prev_speed - veh.speed) / self.timestep)
            self.add_data(veh, 'target_accel_no_noise_no_failsafe', veh.accel_no_noise_no_failsafe)
            self.add_data(veh, 'target_accel_with_noise_no_failsafe', veh.accel_with_noise_no_failsafe)
            self.add_data(veh, 'target_accel_no_noise_with_failsafe', veh.accel_no_noise_with_failsafe)
            self.add_data(veh, 'vdes', veh.fs.v_des if hasattr(veh, 'fs') else -1)
