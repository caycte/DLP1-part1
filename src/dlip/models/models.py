import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, input_size = 16*16, channels=[1,16,16,32,32]):
        super(Model, self).__init__()
        # We allocate space for the weights
        self.channels = channels

        n = input_size * channels[-1]
        self.conv = nn.Sequential(*self.make_conv_layers())
        self.l1 = nn.Sequential( nn.Linear( n, 512 ), nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear( 512, 10))

    def make_conv_layers(self):
        layers = []

        for i,channel in enumerate(self.channels[1:]):
            in_channel = self.channels[i]
            layers += [nn.Conv2d(in_channel, channel,kernel_size=3,padding=1),nn.ReLU(inplace=True)]
           
        return layers
        
    def forward(self, inputs): 
        conv = self.conv(inputs).view(inputs.shape[0],-1)
        conv = conv.view(conv.size(0), -1)
        h = self.l1(conv)
  
        return F.softmax(h, dim=1)
    
    
        

def load_model(path_checkpoint, modelClass: torch.nn.Module, **kwargs):
    model = modelClass(**kwargs)
    checkpoint = torch.load(path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


def save_model(path_checkpoint, model):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        path_checkpoint,
    )