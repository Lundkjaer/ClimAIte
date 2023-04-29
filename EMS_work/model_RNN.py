import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size_flats, hidden_size1, hidden_size2, output_size, fsight_len, solar_included=True):
        super().__init__()

        # RNNs into linears
        # input = rnn + rnn + raw from forward x linear 1
        self.rnn_size = 32
        self.fsight_len = fsight_len
        self.solar_included = solar_included
        self.flats_len = input_size_flats

        self.rnn_work = nn.RNN(1, self.rnn_size, num_layers=1, batch_first=True)
        if self.solar_included == True:
            self.rnn = nn.RNN(3, self.rnn_size, num_layers=1, batch_first=True)
        else:
            self.rnn = nn.RNN(1, self.rnn_size, num_layers=1, batch_first=True)

        self.linear1 = nn.Linear(self.flats_len + 2*self.rnn_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)
        # self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        # self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        # self.linear5 = nn.Linear(hidden_size4, hidden_size5)
        # self.linear6 = nn.Linear(hidden_size5, output_size)

        

    def forward(self, x): # This is called when calling self.model(state [, state[1] optional])

        if len(x.size()) != 2:
            x = torch.unsqueeze(x, 0)
        assert len(x.size()) == 2, 'Forward definition, the unsqueeze to standardize tensor failed'
        # x1 RNN(x[12:23])
        # x2 linear1(x[23:])
        # hx = self.init_hidden(x.size(0), 4 if self.solar_included else 2) if hx is None else hx
       
        # batch_size = x.size(0)

        if self.solar_included == True: # work hours, temps, rad glo, rad dif, flats
            xs = x.split_with_sizes([72,self.fsight_len,self.fsight_len,self.fsight_len,self.flats_len], 1)
            xwork, xtemps, xradglo, xraddif, xflats = xs[0], xs[1], xs[2], xs[3], xs[4]

            xseq = torch.stack((xtemps, xradglo, xraddif), dim=-1)
        else:  # work hours, temps, flats
            xs = x.split_with_sizes([72,self.fsight_len,self.flats_len], 1)
            xwork, xtemps, xflats = xs[0], xs[1], xs[2]

            xseq = xtemps.unsqueeze(-1)
            
        # sequence_length = self.fsight_len # x.size(1) # , unique for each, separate tensors
        # work_hours = x[-1][72] # , no maintain batches

        xwork = xwork.view(-1, 72, 1)
        outwork, hxwork = self.rnn_work(xwork)
        outwork = outwork[:, -1]

        outseq, hseq = self.rnn(xseq)
        outseq = outseq[:, -1]

        # 
        # x = state_multi.split_with_sizes([3,4,18],1)
        # x1, x2, x3 = x[0], x[1], x[2]
        # new_state_multi = torch.concat((x[0],x[1],x[2]), 1)
        #  list of RNN outputs with flat x inputs in list of list, then flatten

        # combine outputs back into 1 tensor with batch
        x = torch.concat((outwork, outseq, xflats), 1)


        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x) # this was used in tutorial where raw numbers are outputted
        # x = F.sigmoid(self.linear4(x)) # this can be used instead to output float 0-1
        return x
    
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss_record = []
    
    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1: #not the above is not a tuple of multiple, then unsqueeze them into the right format
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )
        
        # 1: predicted Q values with current state
        pred = self.model(state) # gives y values (output layer) from network


        with torch.no_grad():
            target = pred.clone()

            for idx in range(len(game_over)):
                Q_new = reward[idx]
                if not game_over[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                
                target[idx][torch.argmax(action[idx]).item()] = Q_new


        # 2: Q_new = r + y (gamma) * max(next_predicted Q value) # gets only one highest value -> only do this if not done
        # pred.clone()
        # preds[argmax(action)]
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.loss_record.append(loss.tolist())

        self.optimizer.step()