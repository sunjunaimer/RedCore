import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, out_channels=128, kernel_heights=[3,4,5], dropout=0.5):  # 3,4,5
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        self.embd_size = embd_size
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(
            nn.Linear(len(kernel_heights)*out_channels, 2*embd_size),
            nn.ReLU(inplace=True),
        )

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, frame_x):
        batch_size, seq_len, feat_dim = frame_x.size()
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        max_out3 = self.conv_block(frame_x, self.conv3)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)

        x = embd.view(-1, 2, self.embd_size)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        #z = torch.sigmoid(z)
        return z, mu, log_var
        #return embd