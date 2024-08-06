import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import r2_score
import joblib

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use('TkAgg')
plt.ion()

# load data
mat_data = loadmat('data.mat')
sr_ = np.array(mat_data['sr_'])
n_comb = np.array(mat_data['n_comb'])
APR = np.array(mat_data['APR'])

x = sr_
y = np.column_stack((n_comb, APR))
print(x, y)

# standardization
scaler = StandardScaler()
x = scaler.fit_transform(x)
x = torch.tensor(x).to(torch.float32)
y = torch.tensor(y).to(torch.float32)
print(x)

# data partitioning
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# to device
x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)



# model
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_feature, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_hidden3),
            nn.ReLU(),
            nn.Linear(n_hidden3, n_output)
        )

    def forward(self, output):
        return self.model(output)


# model instantiation
net = Net(1, 10, 10, 10, 2).to(device)
print(net)

# optimizer and loss function
optimizer = optim.SGD(net.parameters(), lr=0.1)
loss_func = nn.MSELoss().to(device)

# 用于记录 R² 值
r2_values = []
epochs = []
max_epochs = 10001

for epoch in range(max_epochs):
    # train
    net.train()
    y_pred = net(x_train)
    loss = loss_func(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # eval
    net.eval()
    with torch.no_grad():
        val_y_pred = net(x_val)
        val_loss = loss_func(val_y_pred, y_val)
        # val_y_pred_numpy = val_y_pred.to("cpu").numpy()
        val_r2 = r2_score(y_val.cpu().numpy(), val_y_pred.cpu().numpy())

    if val_r2 >= 0.95:
        print(f"Early stopping at iteration {epoch}, R^2: {val_r2:.4f}")
        break

    if epoch % 20 == 0:
        print("Epoch: {}, Loss: {:.4f}, Validation R²: {:.4f}".format(epoch, loss.item(), val_r2))

        r2_values.append(val_r2)
        epochs.append(epoch)

        plt.cla()  # 清除当前图形
        plt.plot(epochs, r2_values, marker='o', linestyle='-', color='b')
        plt.xlabel('Epoch')
        plt.ylabel('R² Score')
        plt.title('R² Score over Epochs')
        plt.grid(True)
        plt.ylim(0, 1)  # 设置 y 轴范围
        plt.draw()
        plt.pause(0.1)

plt.ioff()
plt.show(block=True)
print("Training done")

# 在测试集上评估模型
net.eval()
with torch.no_grad():
    test_y_pred = net(x_test)
    test_r2 = r2_score(y_test.cpu().numpy(), test_y_pred.cpu().numpy())
    print(f"Test R²: {test_r2:.4f}")

# 保存标准化器
joblib.dump(scaler, 'scaler.pkl')

# 保存模型
torch.save(net.state_dict(), 'model.pth')

print("Scaler and model saved")

