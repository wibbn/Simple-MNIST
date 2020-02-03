import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import Net

def train(train_loader, model, criterion, optimizer, epochs=10, device=th.device("cuda")):
    model.train()

    for epoch in range(epochs):
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            print(output.shape, target.shape)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch} [{batch_id * len(data)}/{len(train_loader.dataset)}] ({100 * batch_id // len(train_loader)}%)\tLoss: {loss.item()}')

def test(test_loader, model, device=th.device("cuda")):
    model.eval()
    # loss = 0
    correct = 0
    
    with th.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # loss += criterion(output, target)
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    print(f'\nTest Results\n----------------------\nAccuracy: {correct}/{len(test_loader.dataset)} ({100 * correct // len(test_loader.dataset)}%)')

def main():

    params = {
        # 'indoor_size': 5, 
        # 'outdoor_size': 25, 
        # 'input_size': (3, 128, 128), 
        'batch_size': 64,
        'epochs': 10, 
        'lr': 1e-2
    }

    train_loader = th.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=params['batch_size'], 
        shuffle=True)

    test_loader = th.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=params['batch_size'], 
        shuffle=True)

    model = Net().to(th.device("cuda"))
    criterion = F.nll_loss
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    train(train_loader, model, criterion, optimizer)
    test(test_loader, model)

if __name__ == '__main__':
    main()