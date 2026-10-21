import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# from sklearn.metrics import mean_squared_error  # mse
# from sklearn.metrics import mean_absolute_error  # mae
# from sklearn.metrics import mean_absolute_percentage_error # mape
# from sklearn.metrics import r2_score  # R square
import os, math,sys,copy
# from data_prep import *

def data_prep(features,seq):
    filepath = './data/chic_1987_2000.xlsx' # 3117,106 >> (62+61)/2=61
    df = pd.read_excel(filepath, engine='openpyxl')  # outlier: 3117 106>>(62+60)/2=61
    df = df[features]

    dt = df.values
    dt = dt.astype("float32")
    r,c = dt.shape # 5114,4
    dt_scaler= dt[:,0].reshape((r,1)) #(5114,)-> (5114,1)
    scaler = []
    dt_out = np.empty((r,0))
    for i in np.arange(c-1):
        dt_scaler =np.concatenate((dt_scaler,dt[:,i+1].reshape((r,1))),axis=1)  # reshape to (5114,1) for every col
    # print(dt_scaler.shape)
    # print(dt_scaler[:5,:])
    # print(dt_scaler[:,0].shape)
    ##################################### normalization
    # for j in np.arange(c):
    #     scaler.append(MinMaxScaler())
    #     # scaler.append(StandardScaler())
    #     tempt = scaler[j].fit_transform(dt_scaler[:, j].reshape((r, 1)))
    #     dt_out = np.concatenate((dt_out,tempt),axis=1)
    # ##################################### standardization
    for j in np.arange(c):
        scaler.append(StandardScaler())
        tempt = scaler[j].fit_transform(dt_scaler[:, j].reshape((r, 1)))
        dt_out = np.concatenate((dt_out,tempt),axis=1)
    dt_T = dt_out.T

    dataX, dataY = [], []
    ################### impact of the day when consequence occurs is included
    for i in range(r - seq + 1):
        y = dt_T[0,i + seq - 1] # (look_back,1)
        x = dt_T[1:r,i:(i + seq)] # (seq,c-1)
        dataY.append([y])
        dataX.append(x)
    # print(dt_T[:,:7])
    # print(dataX[:3])
    ################### not included, only previous days' exposure are included
    # for i in range(r - seq):
    #     x = dt[i:(i + seq), 1:c]  # (seq,c-1)
    #     y = dt[i + seq, 0]  # (look_back,1)
    #     dataX.append(x)
    #     dataY.append([y])
    dataX, dataY = np.array(dataX), np.array(dataY) # (r-(seq-1),feature,seq) (r-(seq-1),1)
    # print(dataX.shape,dataY.shape)

    r1,c1 = dataY.shape # r = 5114-seq+1
    train_size = int(r1 * 0.7) # proportion for partition
    # test_size = r1 - train_size

    torch_dataX= torch.from_numpy(dataX).type(torch.float32) # (r,seq,cols)
    torch_dataY= torch.from_numpy(dataY).type(torch.float32) # (r,1)

    # train_x = torch_dataX[:train_size,:,:] #(train_size,features,seq)
    # train_y = torch_dataY[:train_size,:]   #(test_size,1)
    # test_x = torch_dataX[train_size:,:,:]
    # test_y = torch_dataY[train_size:,:]
    return torch_dataX, torch_dataY,dataX,dataY,scaler


class Crop(nn.Module):
    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        out = x[:, :, :-self.crop_size].contiguous()
        # print(self.crop_size,x.shape,out.shape)
        return out


class TemporalCasualLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding,
                    'dilation': dilation}

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        # print("Causal layer input and output:",n_inputs, n_outputs)
        self.crop1 = Crop(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1,
                                 self.conv2, self.crop2, self.relu2, self.dropout2)
        # shortcut connect
        self.bias = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.net(x)
        b = x if self.bias is None else self.bias(x)
        return self.relu(y + b)


class TemporalConvolutionNetwork(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout): # kenel_size = 2,dropout=0.2
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels) # 4
        tcl_param = {
            'kernel_size': kernel_size,
            'stride': 1,
            'dropout': dropout
        }
        print(num_levels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            # print(in_ch,out_ch)
            tcl_param['dilation'] = dilation
            tcl = TemporalCasualLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)
        self.network = nn.Sequential(*layers)
        # conv_out_params = {'kernel_size': 2,
        #                    'stride': 1,
        #                    'padding': 0,
        #                    'dilation': 0}
        # self.conv_out = nn.Conv1d(num_channels[-1], 1, kernel_size=1)
        # self.network = nn.Sequential(*layers,self.conv_out)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        # input_size=look_back, num_channels=[10]*4
        super(TCN, self).__init__()
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        # self.linear = nn.Linear(num_channels[-1], output_size)
        # self.linear1 = nn.Linear(25, output_size) # seq to 1
        # self.linear2 = nn.Linear(10, output_size)  # feature/channel to 1
        self.linear = nn.Linear(10*25, output_size)
    def forward(self, x):
        y = self.tcn(x)  # [N,C_out,L_out=L_in] (batch, 10, seq)
        # out = self.linear(y[:, :, -1])
        # out = self.linear1(y) #(batch,10,1)
        # out = self.linear2(out[:,:,-1])
        # out = self.linear2(out.squeeze(-1))
        # out = 1
        out = self.linear(torch.flatten(y,1))
        return out

if __name__=="__main__":
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    from torchsummary import summary
    features = ['death','tempt', 'PM10', 'O3']#[['death','tempt','PM10','O3','SO2', 'CO','NO2']]
    seq = 25
    torch_dataX,torch_dataY,dataX,dataY,scaler = data_prep(features,seq)
    print(torch_dataX.shape,torch_dataY.shape)

    # 划分训练集和测试集
    train_size = int(torch_dataY.shape[0] * 0.8)

    x_train, y_train = torch_dataX[:train_size,:,:],torch_dataY[:train_size,:]
    x_test, y_test= torch_dataX[train_size:,:,:],torch_dataY[train_size:,:]
    print(x_train.size(), y_train.size())
    #################################################
    # training epochs
    epochs = 4000
    # temporal casual layer channels
    channel_sizes = [10] * 4
    # convolution kernel size
    kernel_size = 3
    dropout = .0
    model_params = {
        # 'input_size',C_in
        'input_size': 3,  # features
        # 单步，预测未来一个时刻
        'output_size': 1,
        'num_channels': channel_sizes,
        'kernel_size': kernel_size,
        'dropout': dropout
    }
    model = TCN(**model_params)

    # print(summary(model, input_size=(3, seq), batch_size=5))
    torch.onnx.export(model, torch.rand(5,3, seq), "model.onnx")
    # import netron
    # netron.start('model.onnx')

    # import onnx
    # onnx_ori = "model.onnx"
    # onnx_show = "model_show.onnx"
    # onnx_graph = onnx.load(onnx_ori)
    # estimated_graph = onnx.shape_inference.infer_shapes(onnx_graph)
    # onnx.save(estimated_graph, onnx_show)
    # import netron
    # netron.start('model_show.onnx')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=.005)
    mse_loss = torch.nn.MSELoss()

    best_params = None
    min_val_loss = sys.maxsize

    training_loss = []
    validation_loss = []

    x_val = x_test
    y_val = y_test

    for t in range(epochs):
        prediction = model(x_train)
        loss = mse_loss(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_prediction = model(x_val)
        val_loss = mse_loss(val_prediction, y_val)

        training_loss.append(loss.item())
        validation_loss.append(val_loss.item())

        if val_loss.item() < min_val_loss:
            best_params = copy.deepcopy(model.state_dict())
            min_val_loss = val_loss.item()

        if t % 100 == 0:
            diff = (y_train - prediction).view(-1).abs_().tolist()
            print(f'epoch {t}. train: {round(loss.item(), 4)}, '
                  f'validation: {round(val_loss.item(), 4)}')
    fig,ax = plt.subplots()
    ax.set_title('Training Progress')
    # plt.yscale("log")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.plot(training_loss, label='train')
    ax.plot(validation_loss, label='validation')
    ax.legend()
    plt.savefig('c:/Users/m/Desktop/loss_epoch.png', dpi=300, bbox_inches='tight', transparent=False)

    best_model = TCN(**model_params)
    best_model.eval()
    best_model.load_state_dict(best_params)

    # tcn_prediction = best_model(x_test)
    # data_prediction = tcn_prediction.view(-1).detach().numpy().reshape(-1, 1)
    data_true = scaler[0].inverse_transform(dataY)
    tcn_prediction = best_model(torch_dataX)
    data_prediction_s = tcn_prediction.view(-1).detach().numpy().reshape(-1, 1)
    data_prediction = scaler[0].inverse_transform(data_prediction_s)

    # data_prediction = scaler.inverse_transform(tcn_prediction.view(-1).detach().numpy().reshape(-1, 1))
    # data_true = scaler.inverse_transform(data.reshape(-1, 1))
    # dummy_prediction = Dummy()(x_test)

    # tcn_mse_loss = round(mse_loss(tcn_prediction, y_test).item(), 4)
    # dummy_mse_loss = round(mse_loss(dummy_prediction, y_test).item(), 4)

    # plt.title(f'Test| TCN: {tcn_mse_loss}; Dummy: {dummy_mse_loss}')
    fig1,ax1 = plt.subplots(2,1,sharex=True)
    ax1[0].set_ylabel('Scaled Mortality')
    ax1[0].plot(np.arange(data_prediction_s.size), data_prediction_s, color='red', label='tcn_scale')
    ax1[0].plot(dataY, alpha=0.5, label='real_scale')
    ax1[0].axvline(x=train_size, color='red', linestyle='--', linewidth=2, label='train:test')
    ax1[0].legend()
    # ax1.plot(np.arange(torch_dataY.shape[0]  - len(tcn_prediction), torch_dataY.shape[0] ), data_prediction, color='red', label='tcn')

    ax1[1].set_ylabel('Mortality')
    ax1[1].set_xlabel('Time samples from 1987 to 2000')
    ax1[1].plot(np.arange(data_prediction.size), data_prediction, color='red', label='tcn')
    ax1[1].plot(data_true, alpha=0.5, label='real')
    ax1[1].axvline(x=train_size, color='red', linestyle='--', linewidth=2, label='train:test')
    ax1[1].legend()
    plt.savefig('c:/Users/m/Desktop/prediction_real.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.show()




