import numpy as np

'''
k=0.5, Kp=1.0,  L=2.9
'''

class StanleyPID:

    def __init__(self, k=2.5, Kp=1.0, Ki=0.1, Kd=2.0,  L=2.9):
        self.k = k  # control gain
        self.Kp = Kp  # speed proportional gain
        self.Kd = Kd
        self.Ki = Ki
        self.I = 0
        self.Imax = np.deg2rad(15.0)
        self.L = L  # [m] Wheel base of vehicle

        self.theta_d = 0
        self.theta_e = 0
        self.delta = 0

    def normalize_angle(self, angle):
        if angle > np.pi:
            angle -= 2.0 * np.pi
        if angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def pid_control(self, target, current):
        return self.Kp * (target - current)

    def stanley_control(self, x, y, yaw, v, cx, cy, cyaw, last_target_idx):
        current_target_idx, error_front_axle = self.calc_target_index(x, y, yaw, cx, cy)
        if last_target_idx >= current_target_idx:
            current_target_idx = last_target_idx

        # theta_e corrects the heading error
        theta_e = self.normalize_angle(cyaw[current_target_idx] - yaw)
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.k * error_front_axle, v)
        # Steering control
        delta = theta_e + theta_d

        k_e = self.k / (1 + np.abs(v))
        self.I = np.clip(self.I + self.Ki*error_front_axle, -self.Imax, self.Imax)
        delta = np.arctan(k_e * error_front_axle) + self.Kd*theta_e + self.I
        #print('Integral=', np.rad2deg(self.I))

        self.theta_e = theta_e
        self.theta_d = theta_d
        self.delta = delta

        return delta, current_target_idx

    def calc_target_index(self, x, y, yaw, cx, cy):
        # Calc front axle position
        fx = x + self.L * np.cos(yaw)
        fy = y + self.L * np.sin(yaw)

        # Search nearest point index
        dx = [fx - icx for icx in cx]
        dy = [fy - icy for icy in cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(yaw + np.pi / 2),
                          -np.sin(yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle

    def control(self, x, y, yaw, v, traj, target_speed):
        target_idx = 1
        cx, cy = traj[1, 1:], traj[0, 1:]
        cyaw = np.arctan2(traj[0, 1:] - traj[0, 0:-1], traj[1, 1:] - traj[1, 0:-1])
        acc = self.pid_control(target_speed, v)
        steer, target_idx = self.stanley_control(x, y, yaw, v, cx, cy, cyaw, target_idx)
        return acc, steer, target_idx