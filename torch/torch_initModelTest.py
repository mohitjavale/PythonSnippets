import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 512, bias=True),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 256, bias=True),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 128, bias=True),
            nn.ELU(alpha=1.0),
            nn.Linear(128, 10, bias=True),
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        # self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        torch.set_default_device(self.device)


    def forward(self, x):
        logits = self.net(x)
        return logits
    
    def train(self, x, y):
        self.pred = self.forward(x)
        self.loss = self.loss_fn(self.pred, y)
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        print(self.loss.item())

    def eval(self, x, y):
        self.pred = self.forward(x)
        self.loss = self.loss_fn(self.pred, y)
        print(self.loss.item())
        

if __name__=="__main__":

    for i in range(1000):
        torch.manual_seed(i)
        my_net = NeuralNetwork()
        # my_net.to(my_net.device)

        x = torch.rand((1000,10), device=my_net.device)*1 + torch.ones((1000,10), device=my_net.device)*0
        y = my_net(x)
        print(f'max = {torch.max(y)} ; min = {torch.min(y)}')

