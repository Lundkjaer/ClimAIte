import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.linear5 = nn.Linear(hidden_size4, hidden_size5)
        self.linear6 = nn.Linear(hidden_size5, output_size)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = F.gelu(self.linear2(x))
        x = F.gelu(self.linear3(x))
        x = F.gelu(self.linear4(x))
        x = F.gelu(self.linear5(x))
        x = self.linear6(x) # this was used in tutorial where raw numbers are outputted
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
        pred = self.model(state) # gives 3 values (output layer) from network
        # TODO fix to take many values rather than single argmax

        with torch.no_grad():
            target = pred.clone()

            for idx in range(len(game_over)):
                Q_new = reward[idx]
                if not game_over[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                
                target[idx][torch.argmax(action).item()] = Q_new


        # 2: Q_new = r + y (gamma) * max(next_predicted Q value) # gets only one highest value -> only do this if not done
        # pred.clone()
        # preds[argmax(action)]
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.loss_record.append(loss.tolist())

        self.optimizer.step()