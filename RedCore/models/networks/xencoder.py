#https://debuggercafe.com/getting-started-with-variational-autoencoders-using-pytorch/
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearVXE(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim):
        super(LinearVXE, self).__init__()

        self.feature_dim = feature_dim
        # encoder
        #self.enc1 = nn.Linear(in_features=input_dim, out_features=int(input_dim/2))
        #self.enc2 = nn.Linear(in_features=int(input_dim/2),  out_features=feature_dim*2)

        self.encoder = torch.nn.Sequential(
                      torch.nn.Linear(in_features=input_dim, out_features=int(input_dim/2)),
                      torch.nn.ReLU(),
                      nn.BatchNorm1d(int(input_dim/2)),
                      torch.nn.Linear(in_features=int(input_dim/2), out_features=feature_dim*2),# out_features=int(input_dim/4)),
                      #torch.nn.ReLU(),
                      #nn.BatchNorm1d(int(input_dim/4)),
                      #torch.nn.Linear(in_features=int(input_dim/4),  out_features=feature_dim*2),
                      )
 
        # decoder 
        #self.dec1 = nn.Linear(in_features=feature_dim, out_features=int(output_dim/2))
        #self.dec2 = nn.Linear(in_features=int(output_dim/2), out_features=output_dim)

        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(in_features=feature_dim, out_features=int(output_dim/2)),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=int(output_dim/2), out_features=output_dim),
                #torch.nn.Sigmoid()
                )


    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
        #x = F.relu(self.enc1(x))
        x = self.encoder(x)
        x = x.view(-1, 2, self.feature_dim)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        #x = F.relu(self.dec1(z))
        #reconstruction = torch.sigmoid(self.dec2(x))
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var