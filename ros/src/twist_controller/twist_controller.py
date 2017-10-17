from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, info):
        self.info = info
        self.yaw_controller = YawController(
            self.info.wheel_base,
            self.info.steer_ratio,
            self.info.min_speed,
            self.info.max_lat_accel,
            self.info.max_steer_angle
        )
        self.pid = PID(
            kp=0.1,
            ki=0.1,
            kd=0.1,
            mn=self.info.decel_limit,
            mx=self.info.accel_limit
        )
        self.steering_filter = LowPassFilter(tau=3, ts=1)
        self.acceleration_filter = LowPassFilter(tau=3, ts=1)

    def reset(self):
        self.pid.reset()

    def control(self, twist_cmd, current_velocity, delta_time):
        # Return throttle, brake, steer
        linear_velocity = abs(twist_cmd.twist.linear.x)
        angular_velocity = twist_cmd.twist.angular.z
        current_velocity = current_velocity.twist.linear.x
        velocity_error = linear_velocity - current_velocity

        next_steering = self.yaw_controller.get_steering(
            linear_velocity,
            angular_velocity,
            current_velocity
        )
        next_steering = self.steering_filter.filt(next_steering)

        #next_acceleration = self.pid.step(velocity_error, delta_time)
        next_acceleration = velocity_error / delta_time
        next_acceleration = self.acceleration_filter.filt(next_acceleration)

        if abs(next_acceleration) < self.info.brake_deadband:
            return 0.0, 0.0, next_steering

        torque = next_acceleration * (self.info.vehicle_mass + self.info.fuel_capacity * GAS_DENSITY) * self.info.wheel_radius

        if torque > 0.0:
            throttle = next_acceleration
            brake = 0.0
        else:
            throttle = 0.0
            brake = abs(torque)

        return throttle, brake, next_steering
