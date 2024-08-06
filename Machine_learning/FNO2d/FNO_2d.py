import torch
import matplotlib.pyplot as plt
from data_generation import data_generator
from neuralop.models import FNO
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss, Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %% data_generation
train_loader, test_loaders, data_processor = data_generator(1000,50)
data_processor = data_processor.to(device)
# %%
model = FNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
model = model.to(device)
n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')

# %%
optimizer = torch.optim.Adam(model.parameters(),
                             lr=8e-3,
                             weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# %%
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}

# %%
trainer = Trainer(model=model,
                  n_epochs=200,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)
# %%
trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)

# %%
test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index * 3 + 1)
    ax.imshow(x[0].cpu().numpy(), cmap='gray')
    if index == 0:
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index * 3 + 2)
    ax.imshow(y.squeeze().cpu().numpy())
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index * 3 + 3)
    ax.imshow(out.squeeze().detach().cpu().numpy())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()

