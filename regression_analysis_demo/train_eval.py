#_________________________________________________________
#Code by Ches                                             |
#contact@ches.darkshades@gmail.com || +2349057900367      |
#_________________________________________________________|


import torch, fc_network, os
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split as tts 


#dataloader
data_path = "data/BTC-USD.csv"
test_size = 0.2 # 20% of data
epochs = 100
in_features = 3
out_features = 1
lr = 0.001
batch_size = 100
model_path = "saved_model_path/trader.pth.tar"


def dataloader(data_path = data_path):
    data = pd.read_csv(data_path)
    features_col = ['Open', 'High', 'Low']
    X = data[features_col].values.reshape(-1, 3)/100
    Y = data['Close'].values.reshape(-1, 1)/100

    return (X, Y)


def save_model(model, path = model_path):
    print('saving model >>>>>>>>>>>')
    torch.save(model, path)
    print('saved model successfully >>>>>>>>>>>')


#data splicer
def data_splicing(data = dataloader(), test_size = test_size):
    X_train, X_eval, Y_train, Y_eval = tts(data[0], data[1], test_size=test_size)
    X_train, X_eval, Y_train, Y_eval = (torch.Tensor(X_train), torch.Tensor(X_eval),
                                        torch.Tensor(Y_train), torch.Tensor(Y_eval)
                                    )
    return (X_train, Y_train, X_eval, Y_eval)


#weight init
def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.08, 0.08)
        m.bias.data.fill_(0)


trader = fc_network.trader(in_features, out_features)
trader.apply(init_weight)

optimizer = optim.Adam(trader.parameters(), lr = lr)
criterion = nn.MSELoss()


#visualize loss_vs_epoch trend
def visualize(loss, epoch):
    plt.style.use('bmh')
    plt.title('loss vs epoch')
    plt.plot(loss, epoch)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


#train_process
def train_process(data = data_splicing(),  epochs = epochs, batch_size = batch_size):
    if not os.path.isfile(model_path):
        X = data[0]
        Y = data[1]
        epoch_axis = np.array([])
        epoch_loss = np.array([])
        LOSS = np.array([])

        for epoch in range(epochs):
            epoch_axis = np.append(epoch_axis, epoch)
            for batch in tqdm(range(0, len(X), batch_size)):
                trader.zero_grad()
                batch_X = X[batch:batch + batch_size]
                output = trader(batch_X)
                loss = criterion(output, Y[batch:batch + batch_size])
                loss.backward()
                optimizer.step()
                LOSS = np.append(LOSS, loss.detach())

            epoch_loss = np.append(epoch_loss, np.mean(LOSS))
            print(f'epoch:{epoch+1} \t loss:{np.mean(LOSS)}')
            LOSS = np.array([])
        
        model_state = {'model_state':trader.state_dict(), 
                        'lossFunc':criterion.state_dict(),
                        'optimizer':optimizer.state_dict()
                        }
        save_model(model_state)

        visualize(loss=epoch_loss, epoch=epoch_axis)
    else:
        pass


#evaluate the model on out of saple data after training
def evaluation(data = data_splicing()):
    eval_X = data[2]
    #denormalize targets
    eval_Y = data[3].numpy()*100

    #load model
    model_state = torch.load(model_path)
    trader_model = fc_network.trader(in_features, out_features)
    trader_model.load_state_dict(model_state['model_state'])
    trader_model.eval()

    with torch.no_grad():
        prediction = trader_model(eval_X)
        #denormalize
        prediction  = np.array(prediction.detach())*100

        comparism_data_frame = pd.DataFrame()
        comparism_data_frame['actual'] = eval_Y.reshape(-1)
        comparism_data_frame['predicted'] = prediction.reshape(-1)
        print(comparism_data_frame)

        comparism_data_frame.head(10).plot(kind='bar')
        plt.grid(which='major', linestyle='-', linewidth=0.5, color='green')
        plt.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
        plt.show()


train_process()
evaluation()













