import torch
import torch.nn as nn
import platform


class VehicleResponse(nn.Module):
    """
        A ground vehicle motion simulation. This simulation takes a reference trajectory(ies) as input
        and outputs the car response trajectory(ies). A reference trajectory is composed by a 3d vector with the following components:
        trajectory = [x(t0), y(t0), v(t0)
                      x(t1), y(t1), v(t1)
                      :      :      :
                      :      :      :
                      x(tf), y(tf), v(tf)]
        where x and y are measured in [meters], and v is measured in [meters/second]. Sampling interval is assumed to be uniform and given.
        an output response has similar components with an additional element which is the heading (yaw) of the vehicle at each time step, measured in [deg].

        Attributes:
            vehicle model:
                v (float): reference forward velocity [m/s]
                dt (float): sampling interval [sec]
                L (float): wheel-base [m]
                max_steer (float): total maximum steering angle [deg]
                max_delta_steer (float): maximum steering angle in a single sample [deg]

            low-level controller parameters:
                forward acceleration controller (stanley controller):
                    k (float): stanely controller constant
                    I (float): stanley controller memory
                    Imax (float): stanley controller saturation limit
                steering controller (pid controller):
                    Kp (float): proportional control constant
                    Ki (float): integral control constant
                    Kd (float): derivative control constant

            optimization parmeters:
                batch_size (int): amount of trajectories in each batch
                max_simulation_time (float): simulation termination time [sec]
                time (float): initial time stamp [sec]
                n (int): number of samples in each trajectory
                device (str): optimization device [cpu or gpu]

        Methods:
            init: Initializes class parameters.
            deg_to_rad: converts an angle representation from degrees to radians
            rad_to_rad: converts an angle representation from radians to degrees
            normalize_angle: modulates an angle [rad] into an [0, 2 * pi] range
            update: updates the vehicle state [steering angle, yaw, x, y, v] using motion equations given an acceleration command and steering angle
            pid_control: calculates an acceleration command given a reference velocity and current velocity
            stanley_control: calculates steering angle command given a reference trajectory [x, y, v, yaw] and current trajectory parameters [cx, cy, cyaw]
            calc_target_index: currently unused.
            control: synchronizes steering and acceleration controllers
            forward: returns the vehicle response given reference trajectory(ies)

        """
    def __init__(self, v=0.0, dt=1.0 / 30.0, L=2.9, max_steer=30.0, max_delta_steer=10.0,      # First line- Vehicle init parameters
                       k=2.5, Kp=1, Ki=0.1, Kd=2.0,                                            # Second line- controller init parameters
                       batch_size=32, max_simulation_time=9.0, time=0.0, n=30, device='cpu'):  # Third line- training parameters

        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.pi = 3.14159265359
        self.target_idx = 0
        self.max_simulation_time = max_simulation_time
        self.time = time
        self.dt = dt
        self.n = n
        # self.n = int((self.max_simulation_time - self.time) / self.dt)
        # Vehicle init
        self.max_steer = self.deg_to_rad(max_steer)  # [rad] max steering angle
        # self.max_steer = max_steer  # [rad] max steering angle
        self.max_delta_steer = self.deg_to_rad(max_delta_steer)
        # self.max_delta_steer = max_delta_steer
        self.x = torch.zeros(batch_size, 1, device=self.device, requires_grad=True)
        self.y = torch.zeros(batch_size, 1, device=self.device, requires_grad=True)
        self.yaw = torch.zeros(batch_size, 1, device=self.device, requires_grad=True)
        self.v = (torch.ones(batch_size, 1, device=self.device, requires_grad=True) * v)
        self.true_steer = 0
        self.L = L  # [m] Wheel base of vehicle

        # Stanley controller init
        self.k = k  # control gain
        self.Kp = Kp  # speed proportional gain
        self.Kd = Kd
        self.Ki = Ki
        self.I = 0
        self.Imax = self.deg_to_rad(15.0)
        self.theta_d = 0
        self.theta_e = 0
        self.delta = 0

    def deg_to_rad(self, angle_deg):
        """
            Returns a radians representation of a degrees represented angle.

                    Parameters:
                            angle_deg (tensor): a tensor with angles in degrees

                    Returns:
                            (tensor): a tensor with angles in radians
        """
        return angle_deg * (self.pi / 180.0)

    def rad_to_rad(self, angle_rad):
        """
            Returns a degrees representation of a radians represented angle.

                    Parameters:
                            angle_rad (tensor): a tensor with angles in radians

                    Returns:
                            (tensor): a tensor with angles in degrees
        """
        return angle_rad * (180.0 / self.pi)

    def normalize_angle(self, angles):
        """
            Modulates an input angles tensor into a [0, 2* pi] range.

                    Parameters:
                            angles (tensor): a tensor with angles in radians

                    Returns:
                            (tensor): a tensor with angles in radians in [0, 2 * pi] range
        """
        idx_high = angles > 2 * self.pi
        idx_low = angles < -2 * self.pi
        angles[idx_high] -= 2 * self.pi
        angles[idx_low] += 2 * self.pi
        return angles

    def update(self, acceleration, delta):
        """
            updates the vehicle trajectory parameters.

                    Parameters:
                            acceleration (tensor): acceleration command
                            delta (tensor): steering angle command

                    Returns:
                            none

                    Additional info:
                        vehicle trajectory parameters are updated using class members [true_steer, yaw, x, y, v]
        """
        delta = torch.clip(delta, -self.max_steer, self.max_steer)
        delta_steer = torch.clip(delta - self.true_steer, -self.max_delta_steer, self.max_delta_steer)
        self.true_steer = self.true_steer + delta_steer
        self.true_steer = torch.clip(self.true_steer, -self.max_steer, self.max_steer)
        self.yaw = self.yaw + (self.v / self.L) * torch.tan(self.true_steer) * self.dt
        self.yaw = self.normalize_angle(self.yaw)
        self.x = self.x + self.v * torch.cos(self.yaw) * self.dt
        self.y = self.y + self.v * torch.sin(self.yaw) * self.dt
        self.v = self.v + acceleration * self.dt
        # self.target_idx += 1
        return

    def pid_control(self, target, current):
        """
            calculates an acceleration command.

                    Parameters:
                            target (tensor): target velocity
                            current (tensor): current velocity

                    Returns:
                            (tensor): acceleration command
        """
        tracking_error = target - current
        # return self.Kp * tracking_error + 1 * self.Kd * tracking_error / self.dt
        return self.Kp * tracking_error

    def stanley_control(self, x, y, yaw, v,                                             # Vehicle state?
                              cx, cy, cyaw, last_target_idx):                           # Reference state?
        """
            calculates steering angle command.

                    Parameters:
                            x (tensor): target trajectory x values.
                            y (tensor): target trajectory y values.
                            yaw (tensor): target trajectory yaw values. (mostly unused)
                            v (tensor): target trajectory velocity values.

                            cx (tensor): current trajectory x value.
                            cy (tensor): current trajectory y value.
                            cyaw (tensor): current trajectory yaw value
                            last_target_idx (tensor): previous closest point on the reference trajectory in the y direction.
                    Returns:
                            delta (tensor): steering command
                            current_target_idx (tensor): current closest point on the reference trajectory in the y direction.
        """
        current_target_idx, error_front_axle = self.calc_target_index(x, y, yaw, cx, cy)
        if last_target_idx >= current_target_idx:
            current_target_idx = last_target_idx
        # theta_e corrects the heading error
        theta_e = self.normalize_angle(cyaw[:, current_target_idx] - torch.squeeze(yaw))
        # theta_d corrects the cross track error
        # theta_d = torch.arctan2(self.k * error_front_axle, v).squeeze()
        # Steering control
        # delta = theta_e + theta_d
        k_e = (self.k / (1 + torch.abs(v)))
        self.I = torch.clip(self.I + self.Ki * error_front_axle, -self.Imax, self.Imax)
        delta = torch.arctan(k_e * error_front_axle) + self.Kd * torch.unsqueeze(theta_e, dim=1) + self.I

        self.theta_e = theta_e
        # self.theta_d = theta_d
        self.delta = delta
        current_target_idx = self.target_idx
        return delta, current_target_idx

    def calc_target_index(self, x, y, yaw, cx, cy):
        """
           calculates current closest point on the reference trajectory in the y direction, and front axle distance from that point

                   Parameters:
                           x (tensor): target trajectory x values
                           y (tensor): target trajectory y values
                           yaw (tensor): target trajectory yaw values. (mostly unused)

                           cx (tensor): current trajectory x value
                           cy (tensor): current trajectory y value

                   Returns:
                           self.target_idx (tensor): current closest point on the reference trajectory in the y direction
                           error_front_axle (tensor): and front axle distance from y(self.target_idx)
       """
        # Calc front axle position
        fx = x + self.L * torch.cos(yaw)
        fy = y + self.L * torch.sin(yaw)
        # Search nearest point index
        dx = fx - cx
        dy = fy - cy
        # d = torch.hypot(dx, dy)
        # target_idx = torch.argmin(d, dim=1)
        # Project RMS error onto front axle vector
        front_axle_vec = torch.hstack((-torch.cos(yaw + self.pi / 2), -torch.sin(yaw + self.pi / 2))).float()
        dxdy = torch.transpose(torch.vstack((dx[:, self.target_idx], dy[:, self.target_idx])), 0, 1)
        error_front_axle = torch.sum((dxdy * front_axle_vec), dim=1, keepdim=True)
        return self.target_idx, error_front_axle

    def control(self, x, y, yaw, v, traj, target_speed):
        """
           calculates acceleration and steering commands

                   Parameters:
                           x (tensor): target trajectory x values
                           y (tensor): target trajectory y values
                           yaw (tensor): target trajectory yaw values. (mostly unused)
                           target_speed (tensor): target trajectory velocity values

                           traj (tensor): current trajectory x and y values
                           v (tensor): current velocity

                   Returns:
                           acc (tensor): acceleration command
                           steer (tensor): steering command
                           self.target_idx (tensor): currently unused
       """
        cx, cy = traj[:, 1, 1:], traj[:, 0, 1:]
        if platform.system() == 'Linux':
            cyaw = torch.arctan((traj[:, 0, 1:] - traj[:, 0, 0:-1]) / (traj[:, 1, 1:] - traj[:, 1, 0:-1]))
        else:
            cyaw = torch.arctan2(traj[:, 0, 1:] - traj[:, 0, 0:-1], traj[:, 1, 1:] - traj[:, 1, 0:-1])
        acc = self.pid_control(target_speed, v)
        steer, self.target_idx = self.stanley_control(x, y, yaw, v, cx, cy, cyaw, self.target_idx)
        return acc, steer, self.target_idx

    def forward(self, trajectories_batch, target_speed):
        """
           calculates vehicle response to reference trajectory(ies)

                   Parameters:
                           trajectories_batch (tensor): target trajectory x and y values
                           target_speed (tensor): target velocity values

                   Returns:
                           x (tensor): vehicle response, x components
                           y (tensor): vehicle response, y components
                           v (tensor): vehicle response, forward velocity components
                           yaw (tensor): vehicle response, heading components
                           self.target_idx (tensor): currently unused
       """
        x = self.x
        y = self.y
        v = self.v
        yaw = self.yaw
        while self.target_idx < self.n - 1:
            acc, steer, target_idx = self.control(self.x,
                                                  self.y,
                                                  self.yaw,
                                                  self.v,
                                                  trajectories_batch,
                                                  target_speed)
            # x_update, y_update, v_update, yaw_update = self.update(acc, steer)
            # x = torch.hstack((x, x_update))
            # y = torch.hstack((y, y_update))
            # v = torch.hstack((v, v_update))
            # yaw = torch.hstack((yaw, yaw_update))
            self.target_idx += 1
            self.time += self.dt
            self.update(acc, steer)
            x = torch.hstack((x, self.x))
            y = torch.hstack((y, self.y))
            v = torch.hstack((v, self.v))
            yaw = torch.hstack((yaw, self.yaw))

        return x, y, v, yaw, self.target_idx
