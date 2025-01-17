# from models.basic.loss import VFELossTotal
import torch
from nets.sequential_critic_nets import ValueNet

# Assuming VFELossTotal is not used, as it wasn't present in the original code.

env = "ant"
model = "validationround"
seed = 1
eval_episodes = 10
test_episodes = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PerformanceEval:
    def __init__(self, env, model, seed, eval_episodes, epochs=3):
        self.env = env
        self.model = model
        self.seed = seed
        self.device = device
        self.eval_episodes = eval_episodes
        self.test_episodes = test_episodes
        self.gamma = 0.995
        self.learning_rate = 3e-4
        self.epochs = epochs  # Number of epochs for training
        self.data = self.get_data()
        self.values = self.value_tupples()
        self.G = self.get_G(self.values)
        self.loss_fn = torch.nn.MSELoss()
        self.init_value_net()
        self.optim_model = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate
        )
        self.train()

    def load_path(self, i):
        path_i = (
            f"_logs/{self.env}/{self.model}/seed_0{self.seed}/performance_infos_{i}.pt"
        )
        return torch.load(path_i)

    def get_data(self):
        data = []
        for i in range(self.test_episodes):
            data.append(self.load_path(i))
        return data

    def value_tupples(self):
        values = [[] for _ in range(self.eval_episodes)]
        for i in range(self.eval_episodes):
            for j in range(len(self.data[i]["trajectory"])):
                s, r = (
                    self.data[i]["trajectory"][j][2],
                    self.data[i]["trajectory"][j][5],
                )
                values[i].append((s, r))
        return values

    def test_tupples(self):
        values_test = [[] for _ in range(self.eval_episodes, self.test_episodes)]
        for i in range(self.eval_episodes, self.test_episodes):
            discounted_reward = self.data[i]["episode_info"][0][1]
            for j in range(len(self.data[i]["trajectory"])):
                s, r = (
                    self.data[i]["trajectory"][j][2],
                    self.data[i]["trajectory"][j][5],
                )
                values_test[i - self.eval_episodes].append((s, r, discounted_reward))
        return values_test

    def get_G(self, values):  # return (s, G)
        G = [[] for _ in range(self.eval_episodes)]
        for i in range(len(values)):
            for j in range(len(values[i]) - 1, -1, -1):  # Reverse iteration
                s, r = values[i][j]
                if j == len(values[i]) - 1:
                    G[i].append((s, r))
                else:
                    G[i].append((s, r + self.gamma * G[i][-1][1]))
        return G

    def init_value_net(self):
        # Initialize the ValueNet
        s, _ = self.G[0][0]  # Extract the first state tensor
        s = s.unsqueeze(0) if s.ndim == 1 else s  # Ensure batch dimension
        self.net = ValueNet((s.size(1),)).to(self.device)

    def train(self):
        for epoch in range(self.epochs):  # Outer loop for epochs
            epoch_loss = 0.0
            for i in range(len(self.G)):
                for j in range(len(self.G[i])):
                    s, g = self.G[i][j]
                    s = s.unsqueeze(0) if s.ndim == 1 else s  # Ensure batch dimension
                    s = s.to(self.device, dtype=torch.float32)
                    g = torch.tensor(g, dtype=torch.float32, device=self.device)
                    v = self.net(s)  # Predict value for state `s`
                    loss = self.update(v, g)
                    epoch_loss += loss.item()  # Accumulate loss for reporting
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss / len(self.G)}")

    def update(self, v, g):
        self.optim_model.zero_grad()
        loss = self.loss_fn(v, g.view_as(v))
        # loss = (v-g)**2
        loss.backward()
        self.optim_model.step()
        return loss

    def predict(self):
        # Predict values for test data
        predic_list = []
        true_list = []
        test_tupples = self.test_tupples()
        for i in range(len(test_tupples)):
            s, _, discounted_reward = test_tupples[i][0]
            s = s.unsqueeze(0) if s.ndim == 1 else s
            s = s.to(self.device, dtype=torch.float32)
            predic_list.append(self.net(s).item())  # Convert prediction to scalar
            true_list.append(discounted_reward)
        print(sum(predic_list) / len(predic_list), sum(true_list) / len(true_list))


model = PerformanceEval(env, model, seed, eval_episodes)
model.predict()
