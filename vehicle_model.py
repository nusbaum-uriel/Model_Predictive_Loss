import numpy as np
from matplotlib import pyplot as plt
from vehicle_response import VehicleResponse
import torch
import torch.nn as nn
import torch.optim as optim
import timeit

if __name__ == "__main__":
    init_st = timeit.default_timer()
    # Trajectories parameters
    max_simulation_time = 9.0
    time = 0.0
    dt = 1.0 / 30.0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # n = 30
    trajectory_length = 20
    n = int((max_simulation_time - time) / dt)
    # c_x_mean = np.linspace(-1, 7, n)
    c_x_variance = 0.15
    x_min = np.random.normal(-5, c_x_variance)
    x_max = np.random.normal(7, c_x_variance)
    x_trend = np.linspace(x_min, x_max, n)
    c_x = np.random.normal(0, c_x_variance, n)
    c_x_mean = x_trend + c_x
    c_y_mean = np.linspace(0, trajectory_length, n)
    c_y_variance = 0

    # making a target trajectories batch
    batch_size = 32
    target_trajectories = np.empty(shape=(batch_size, 2, n))
    for i in range(0, batch_size):
        target_trajectories[i, 0:2, 0:n] = np.concatenate((np.random.normal(loc=c_x_mean.reshape(1, n), scale=c_x_variance, size=(1, n)),
                                                    np.random.normal(loc=c_y_mean.reshape(1, n), scale=c_y_variance, size=(1, n))), axis=0)
    # target_trajectories = nn.Parameter(torch.tensor(trajectories, device=device, requires_grad=True))
    target_trajectories = torch.tensor(target_trajectories, device=device)
    # targets index batch
    target_speed = torch.ones((batch_size, 1)).to(device)
    target_velocity = (trajectory_length / max_simulation_time)
    target_speed = target_speed * target_velocity  # [m/s]

    # Trajectories to be optimized
    class Trajectories(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.trajectories = nn.Parameter(torch.randn(target_trajectories.size(), dtype=torch.double, device=device, requires_grad=True))

    trajectories = Trajectories()

    # Create the control + sim module
    vehicle_response = VehicleResponse(v0=0.0, dt=1.0 / 30.0, L=2.9, max_steer=30.0, max_delta_steer=10.0,
                                       k=2.5, Kp=1, Ki=0.1, Kd=1.0,
                                       max_simulation_time=9.0, initial_time=0.0, n=n, device=device)
    init_end = timeit.default_timer()
    # print('Initiation time:', init_end - init_st, 'seconds')
    st = timeit.default_timer()
    x, y, v, yaw, target_idx = vehicle_response.forward(target_trajectories, target_speed)

    et = timeit.default_timer()
    # responses = torch.transpose(torch.cat((torch.unsqueeze(x, dim=2), torch.unsqueeze(y, dim=2), torch.unsqueeze(v, dim=2), torch.unsqueeze(yaw, dim=2)), dim=2), dim0=1, dim1=2)
    # responses_for_loss = torch.tensor(torch.transpose(torch.cat((torch.unsqueeze(x, dim=2), torch.unsqueeze(y, dim=2)), dim=2), dim0=1, dim1=2), requires_grad=True)
    responses_for_loss = torch.transpose(torch.cat((torch.unsqueeze(y, dim=2), torch.unsqueeze(x, dim=2)), dim=2), dim0=1, dim1=2)
    loss = nn.MSELoss(reduction='sum')
    optimizer = optim.SGD(trajectories.parameters(), lr=0.005, momentum=0.9)

    running_loss = 0
    idx = np.random.randint(0, batch_size)
    for i in range(31):
        optimizer.zero_grad()
        # x, y, v, yaw, target_idx = vehicle_response.forward(target_trajectories, target_speed)
        # responses_for_loss = torch.transpose(torch.cat((torch.unsqueeze(y, dim=2), torch.unsqueeze(x, dim=2)), dim=2), dim0=1, dim1=2)
        # output = loss(responses_for_loss, trajectories.trajectories)
        output = loss(target_trajectories, trajectories.trajectories)
        torch.autograd.backward(output, inputs=trajectories.trajectories)
        # output.backward()
        optimizer.step()
        if i % 5 == 0:
            # fig = plt.figure()
            # # idx = np.random.randint(0, batch_size)
            # plt.plot(y[idx, :].detach().cpu().numpy(), x[idx, :].detach().cpu().numpy(), 'b-.', label='Actual')
            # plt.plot(target_trajectories[idx, 0, :].detach().cpu().numpy(), target_trajectories[idx, 1, :].detach().cpu().numpy(), color='r', linewidth=2, label='Reference')
            # plt.plot(trajectories.trajectories[idx, 0, :].detach().cpu().numpy(), trajectories.trajectories[idx, 1, :].detach().cpu().numpy(), '.', color='g', markersize=2,
            #          label='Optimized')
            x_new, y_new, _, _, _ = vehicle_response.forward(trajectories.trajectories, target_speed)
            # plt.plot(y_new[idx, :].detach().cpu().numpy(), x_new[idx, :].detach().cpu().numpy(), 'k--', label='Optimized response')
            # legend = plt.legend(loc='upper left', shadow=True, fontsize='small')
            # plt.scatter(-10, 0)
            # plt.scatter(10, 0)
            # plt.gca().set_aspect('equal', adjustable='box')
            # plt.show()
            # print(output)

    reference_responses = torch.transpose(torch.cat((torch.unsqueeze(y_new, dim=2), torch.unsqueeze(x_new, dim=2)), dim=2), dim0=1, dim1=2)
    loss_new = nn.MSELoss(reduction='sum')
    trajectories_new = Trajectories()
    optimizer_new = optim.SGD(trajectories_new.parameters(), lr=0.001, momentum=0.9)
    for i in range(200):
        optimizer_new.zero_grad()
        x_new, y_new, _, _, _ = vehicle_response.forward(trajectories_new.trajectories, target_speed)
        if i % 2 == 0:
            fig = plt.figure()
            # idx = np.random.randint(0, batch_size)
            plt.plot(reference_responses[idx, 0, :].detach().cpu().numpy(), reference_responses[idx, 1, :].detach().cpu().numpy(), color='k', label='Reference response')
            plt.plot(y_new[idx, :].detach().cpu().numpy(), x_new[idx, :].detach().cpu().numpy(), color='b', label='Actual')
            plt.plot(target_trajectories[idx, 0, :].detach().cpu().numpy(), target_trajectories[idx, 1, :].detach().cpu().numpy(), color='r', label='Reference')
            plt.plot(trajectories_new.trajectories[idx, 0, :].detach().cpu().numpy(), trajectories_new.trajectories[idx, 1, :].detach().cpu().numpy(), '.', color='g', markersize=2,
                     label='Optimized')
            legend = plt.legend(loc='upper left', shadow=True, fontsize='small')
            plt.scatter(-10, 0)
            plt.scatter(10, 0)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()
        responses_for_loss_new = torch.transpose(torch.cat((torch.unsqueeze(y_new, dim=2), torch.unsqueeze(x_new, dim=2)), dim=2), dim0=1, dim1=2)
        # output_new = loss_new(target_trajectories, trajectories_new.trajectories) + loss_new(reference_responses, responses_for_loss_new)
        output_new = loss_new(reference_responses, responses_for_loss_new)
        torch.autograd.backward(output_new, inputs=trajectories_new.trajectories)
        optimizer_new.step()
