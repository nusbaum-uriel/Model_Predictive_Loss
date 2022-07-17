import numpy as np
from matplotlib import pyplot as plt
from VehicleResponse import VehicleResponse
import torch
import torch.nn as nn
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
    c_x_variance = 0.2
    c_x_variance = 0.02
    c_x_mean = np.random.normal(-2, c_x_variance, n)
    c_y_mean = np.linspace(0, trajectory_length, n)
    c_y_variance = 0

    # making a trajectories batch
    batch_size = 32
    trajectories = np.empty(shape=(batch_size, 2, n))
    for i in range(0, batch_size):
        trajectories[i, 0:2, 0:n] = np.concatenate((np.random.normal(loc=c_x_mean.reshape(1, n), scale=c_x_variance, size=(1, n)),
                                                    np.random.normal(loc=c_y_mean.reshape(1, n), scale=c_y_variance, size=(1, n))), axis=0)
    trajectories = nn.Parameter(torch.tensor(trajectories, device=device, requires_grad=True))
    # trajectories = torch.tensor(trajectories).to(device)
    # targets index batch
    target_speed = torch.ones((batch_size, 1)).to(device)
    target_velocity = (trajectory_length / max_simulation_time)
    target_speed = target_speed * target_velocity  # [m/s]

    # Create the control + sim module
    vehicle_response = VehicleResponse(v=0.0, dt=1.0 / 30.0, L=2.9, max_steer=30.0, max_delta_steer=10.0,
                                       k=2.5, Kp=1, Ki=0.1, Kd=1.0,
                                       batch_size=batch_size, max_simulation_time=9.0, time=0.0, n=n, device=device)
    init_end = timeit.default_timer()
    # print('Initiation time:', init_end - init_st, 'seconds')
    st = timeit.default_timer()
    x, y, v, yaw, target_idx = vehicle_response.forward(trajectories, target_speed)
    et = timeit.default_timer()
    # responses = torch.transpose(torch.cat((torch.unsqueeze(x, dim=2), torch.unsqueeze(y, dim=2), torch.unsqueeze(v, dim=2), torch.unsqueeze(yaw, dim=2)), dim=2), dim0=1, dim1=2)
    # responses_for_loss = torch.tensor(torch.transpose(torch.cat((torch.unsqueeze(x, dim=2), torch.unsqueeze(y, dim=2)), dim=2), dim0=1, dim1=2), requires_grad=True)
    responses_for_loss = torch.transpose(torch.cat((torch.unsqueeze(y, dim=2), torch.unsqueeze(x, dim=2)), dim=2), dim0=1, dim1=2)
    # loss = nn.MSELoss(reduction='sum')
    # output = loss(responses_for_loss, trajectories)
    # output.backward()
    # torch.autograd.backward(output, inputs=trajectories)
    # print('MSE loss:\n', float("{:.2f}".format(output)))
    # print('\n')
    # print('Execution time:\n', float("{:.2f}".format(et - st)), 'seconds')
    # print('\n')
    # print('Total time:\n', float("{:.2f}".format(et - init_st)), 'seconds')
    # print('\n')
    # print('Gradient', output.grad)
    # print('\n')
    # # print('Trajectories\n', trajectories)
    # print('Vehicle response\n ', responses_for_loss)
    # print('Error vector\n', trajectories - responses_for_loss)
    # print('Error sum\n', torch.sum((trajectories - responses_for_loss)**2, dim=2))  # for vector
    # print('Error sum\n', torch.sum((trajectories - responses_for_loss) ** 2))

    fig = plt.figure()
    idx = np.random.randint(0, batch_size)
    plt.plot(y[idx, :].detach().cpu().numpy(), x[idx, :].detach().cpu().numpy(), color='b', label='Actual')
    plt.plot(y[idx, :].detach().cpu().numpy(), x[idx, :].detach().cpu().numpy(), color='g', label='Predicted')
    plt.plot(trajectories[idx, 0, :].detach().cpu().numpy(), trajectories[idx, 1, :].detach().cpu().numpy(), color='r', label='Reference')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
    plt.scatter(-10, 0)
    plt.scatter(10, 0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

