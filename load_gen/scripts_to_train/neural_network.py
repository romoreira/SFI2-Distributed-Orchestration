import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


class linearRegression(nn.Module): # all the dependencies from torch will be given to this class [parent class] # nn.Module contains all the building block of neural networks:
  def __init__(self,input_dim):
    super(linearRegression,self).__init__()  # building connection with parent and child classes
    self.fc1=nn.Linear(input_dim,10)          # hidden layer 1
    self.fc2=nn.Linear(10,5)                  # hidden layer 2
    self.fc3=nn.Linear(5,3)                   # hidden layer 3
    self.fc4=nn.Linear(3,1)                   # last layer

  def forward(self,d):
    out=torch.relu(self.fc1(d))              # input * weights + bias for layer 1
    out=torch.relu(self.fc2(out))            # input * weights + bias for layer 2
    out=torch.relu(self.fc3(out))            # input * weights + bias for layer 3
    out=self.fc4(out)                        # input * weights + bias for last layer
    return out                               # final outcome




df = pd.read_csv('../reports_to_merge/read_sinusoidal/arquivo_final.csv')



colunas_para_dropar = ["type total", "src_port", "dst_port","Src IP", "Src Port", "Dst IP", "Dst Port", "Label", "pk/s", "row/s", "med", ".95", ".99", ".999", "max"]
y = df['mean']
X = df.drop(columns=colunas_para_dropar)
X = X.drop(columns=['mean'])


# train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True)

# Convertendo para tensores do PyTorch após a divisão dos dados
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

input_dim=X_train.shape[1]
torch.manual_seed(42)  # to make initilized weights stable:
model=linearRegression(input_dim)
loss=nn.MSELoss() # loss function
optimizer = optim.SGD(model.parameters(), lr=0.0001)



num_of_epochs=100
# Listas para armazenar os valores reais e preditos
real_values = []
predicted_values = []
losses = []

# Loop de treinamento
for epoch in tqdm.tqdm(range(num_of_epochs), desc="Training Progress"):
    # give the input data to the architecture
    y_train_prediction = model(X_train)  # model initializing
    loss_value = loss(y_train_prediction.squeeze(), y_train)  # find the loss function
    optimizer.zero_grad()  # make gradients zero for every iteration
    loss_value.backward()  # back propagation
    optimizer.step()  # update weights in NN

    # Armazenar os valores reais e preditos
    real_values.extend(y_train.cpu().detach().numpy())  # Supondo que y_train é um tensor PyTorch
    predicted_values.extend(
        y_train_prediction.cpu().detach().numpy())  # Supondo que y_train_prediction é um tensor PyTorch

    # Armazenar o valor da perda
    losses.append(loss_value.item())

# Calcular MSE e RMSE
mse = ((np.array(real_values) - np.array(predicted_values)) ** 2).mean()
rmse = np.sqrt(mse)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Criar o gráfico da perda
plt.plot(range(num_of_epochs), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()