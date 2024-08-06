import joblib
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载保存的模型
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

# 加载标准化器
scaler_loaded = joblib.load('scaler.pkl')

# 加载模型
model_loaded = Net(1, 10, 10, 10, 2)
model_loaded.load_state_dict(torch.load('model.pth'))
model_loaded.eval()

print("Scaler and model loaded")

# 格式化输入
new_x = torch.tensor([0.7]).reshape(1, 1)
print(new_x)
new_x = torch.tensor(scaler_loaded.transform(new_x),dtype=torch.float32)

# 计算结果
with torch.no_grad():
    new_y_pred1 = model_loaded(new_x)

print(new_y_pred1)
