import torch
from torch.utils.data import DataLoader



# Calculate the accuracy to evaluate the model
def accuracy(dataset, model: torch.nn.Module, device = 'cpu'):

    with torch.no_grad():
        correct = 0
        total = 0
        dataloader = DataLoader(dataset)
        for images, labels in dataloader:
            # images = images.view(-1, 16*16)
            labels = labels.to(device='cpu')
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)  
            correct += (predicted.cpu() == labels).sum()
    
    return correct.item()

    print('Accuracy of the model : {:.2f} %'.format(100*correct.item()/ len(dataset)))
