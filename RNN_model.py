
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, RNN_embed_dim=99, h_RNN_layers=2, h_RNN=72, h_FC_dim=64, drop_p=0.2, num_classes=9, num_heads=8):
        super(RNN, self).__init__()

        self.RNN_input_size = RNN_embed_dim
        self.h_RNN_layers = h_RNN_layers
        self.h_RNN = h_RNN
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.h_RNN, num_heads=self.num_heads, batch_first=True)

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        attention_out, _ = self.multihead_attention(RNN_out, RNN_out, RNN_out)
        # FC layers
        x = self.fc1(attention_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x