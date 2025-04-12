import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self,
                 num_genes,
                 hidden_size=128,
                 dropout=0,
                 CNN=False):
        super().__init__()
        if CNN:
            pass
        else:
            self.encoder = nn.Sequential(
                    nn.Linear(num_genes, hidden_size * 2),
                    nn.LayerNorm(hidden_size * 2),
                    nn.Mish(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.Mish(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size)
                )
    
    def forward(self, x):
        return self.encoder(x)
    
class BaseDecoder(nn.Module):
    def __init__(self,
                 num_genes,
                 hidden_size=128,
                 dropout=0,
                 CNN=False):
        super().__init__()
        if CNN:
            pass
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Mish(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.Mish(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, num_genes)
            )

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self,
                 num_genes,
                 hidden_size=128,
                 dropout=0,
                 masked_weights=0.75,
                 CNN=False,
                 m=0.999):
        super().__init__()

        self.num_genes = num_genes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.masked_weights = masked_weights
        self.m = m

        self.encoderS = BaseEncoder(num_genes=num_genes, 
                                    hidden_size=hidden_size, 
                                    dropout=dropout, 
                                    CNN=CNN)
        self.encoderT = BaseEncoder(num_genes=num_genes, 
                                    hidden_size=hidden_size, 
                                    dropout=dropout, 
                                    CNN=CNN)
        for param in self.encoderT.parameters():
            param.requires_grad = False
        
        self.maskPredictor = nn.Linear(hidden_size, num_genes)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + num_genes, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(inplace=True)
        )

        self.decoder = BaseDecoder(num_genes=num_genes, 
                                   hidden_size=hidden_size, 
                                   dropout=dropout, 
                                   CNN=CNN)
        
        self.weight_r = nn.Parameter(torch.zeros(1))
        self.weight_m = nn.Parameter(torch.zeros(1))
        self.weight_l = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def _update_teacher(self):
        for param_S, param_T in zip(self.encoderS.parameters(), self.encoderT.parameters()):
            param_T.data = self.m * param_T.data + (1 - self.m) * param_S.data

    def forward(self, x, comp_x):
        latent_S = self.encoderS(x)
        mask_pred = self.maskPredictor(latent_S)
        latent_S = self.fusion(torch.cat([latent_S, mask_pred], dim=1))
        reconstruction = self.decoder(latent_S)

        with torch.no_grad():
            # self._update_teacher()
            latent_T = self.encoderT(comp_x)
            latent_T = self.fusion(torch.cat([latent_T, 1 - mask_pred], dim=1))

        return {
            "reconstruction": reconstruction,
            "mask_pred": mask_pred,
            "latent_S": latent_S,
            "latent_T": latent_T
        }
    
    def compute_loss(self, x, y, comp_x, mask, weight_r = 0.3, weight_m = 0.4, weight_l=0.3, UWL=False):
        outputs = self(x, comp_x)
        w_nums = mask * self.masked_weights + (1 - mask) * (1 - self.masked_weights)
        reconstruction_loss = (torch.mul(w_nums, mse(outputs["reconstruction"], y, reduction='none'))).mean()

        mask_loss = bce_logits(outputs["mask_pred"], mask)

        latent_S = F.normalize(outputs["latent_S"], dim=1)
        latent_T = F.normalize(outputs["latent_T"], dim=1)
        latent_loss = 1 - torch.cosine_similarity(latent_S, latent_T, dim=1).mean()
        if UWL:
            weight_r = torch.clamp(self.weight_r, min=-5, max=5)
            weight_m = torch.clamp(self.weight_m, min=-5, max=5)
            weight_l = torch.clamp(self.weight_l, min=-5, max=5)

            precision_l = torch.exp(-weight_l)
            precision_m = torch.exp(-weight_m)
            precision_r = torch.exp(-weight_r)

            weighted_latent_loss = precision_l * latent_loss + precision_l/2
            weighted_mask_loss = precision_m * mask_loss + precision_m/2
            weighted_reconstruction_loss = precision_r * reconstruction_loss + precision_r/2

            loss = weighted_reconstruction_loss + weighted_latent_loss + weighted_mask_loss
            
        else:
            loss = weight_r * reconstruction_loss + weight_m * mask_loss + weight_l * latent_loss
        
        return {
            "total_loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "latent_loss": latent_loss,
            "mask_loss": mask_loss,
            "weight_r": weight_r if not UWL else precision_r.item(),
            "weight_m": weight_m if not UWL else precision_m.item(),
            "weight_l": weight_l if not UWL else precision_l.item()
        }

