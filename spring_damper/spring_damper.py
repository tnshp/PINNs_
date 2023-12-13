from model import DNN, CustomLoss
import os 
import torch
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np


PATH  = 'spring_damper'
DEVICE  = 'cuda'
LOAD_FROM_FILE = True
SAVE_MODEL =  True
ADD_NOISE =  False
save_file = os.path.join(PATH, 'Model_parameters\model1.pt')
if ADD_NOISE:
    save_file = os.path.join(PATH, 'Model_parameters\model1_noise.pt')
num_epochs = 5000

torch.set_default_dtype(torch.float32)


model = DNN([1,128,128,128,128,128, 128, 128, 128, 1]).to(DEVICE)


f = open(os.path.join(PATH,'configs.json'))
data = json.load(f)

x0 = data['initial_values']['position']
v0 = data['initial_values']['velocity']
a0 = data['initial_values']['acceleration']

# Constants
mass = data['constants']['mass']
k = data['constants']['k'] # spring coefficient
b = data['constants']['b'] # damping coefficient

if __name__ == '__main__':
    filename  = os.path.join(PATH, "DATA\data+2.5+2+0.3.csv")
    df = pd.read_csv(filename)   #tested

    x_train = df.iloc[:,0].to_numpy()
    print(x_train.shape)
    if ADD_NOISE:
        noise = np.random.rand(len(x_train))
        x_train = x_train + noise

    x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(DEVICE)
    t_train = torch.tensor(df.iloc[:,4].to_numpy(), dtype=torch.float32, requires_grad=True).to(DEVICE)

    t_train = torch.reshape(t_train, (len(t_train), 1))
    x_train = torch.reshape(x_train, (len(x_train), 1))

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = CustomLoss(mass, k, b)

    if LOAD_FROM_FILE == True:
        model.load_state_dict(torch.load(save_file))

    for epoch in range(num_epochs):
        # Forward pass
        
        x_pred = model(t_train)
        loss = loss_fn.forward(t_train, x_pred, x_train)
        print("Epoch: ", epoch, "Loss: ", loss)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #saving parms

    if SAVE_MODEL:
        torch.save(model.state_dict(), save_file)
    #test data
    t_test = torch.linspace(0, 40, steps = 80).reshape(80,1).to(DEVICE)
    x_pred = model.forward(t_test)

    print(t_test.shape)
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data

    ax.plot(t_test.cpu().detach(), x_pred.cpu().detach(), color = 'red',label = "predicted")

    ax.scatter(t_train.cpu().detach(), x_train.cpu().detach(), label='Data')


    # Add labels and a legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Simple Line Plot')
    ax.legend()

    # Display the plot
    plt.show()
