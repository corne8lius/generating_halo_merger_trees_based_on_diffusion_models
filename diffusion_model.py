import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)          
            x = up(x, t)
        return self.output(x)
    

class Autoencoder_diffusion(nn.Module):
    def __init__(self, nvar, nbr, latent_size):
        super().__init__()
        time_emb_dim = 32
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU())
        
        self.num_branches = nbr
        
        self.first_conv = nn.Conv2d(nvar, 16, kernel_size=(1, 3), stride=1)
        
        self.down = nn.ModuleList([#nn.Conv2d(nvar, 16, kernel_size=(1, 3), stride=1),
                                    nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1),
                                    nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1),
                                    nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1),
                                    nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1)])

        if self.num_branches == 6:
            self.linear_down = nn.Linear(11776, latent_size)
        else:
            self.linear_down = nn.Linear(35328, latent_size)
            
            
        self.last_conv = nn.ConvTranspose2d(16, nvar, kernel_size=(1, 3), stride=1)
        
        self.up = nn.ModuleList([nn.ConvTranspose2d(2 * 256, 128, kernel_size=(3, 1), stride=1),
                                nn.ConvTranspose2d(2 * 128, 64, kernel_size=(3, 1), stride=1),
                                nn.ConvTranspose2d(2 * 64, 32, kernel_size=(3, 1), stride=1),
                                nn.ConvTranspose2d(2 * 32, 16, kernel_size=(1, 3), stride=1),
                                #nn.ConvTranspose2d(2 * 16, nvar, kernel_size=(1, 3), stride=1)
                                ])
            
            
        self.time_emb_down = nn.ModuleList([
                                            #nn.Linear(time_emb_dim, 16),
                                            nn.Linear(time_emb_dim, 32),
                                            nn.Linear(time_emb_dim, 64),
                                            nn.Linear(time_emb_dim, 128),
                                            nn.Linear(time_emb_dim, 256)])
        
        self.time_emb_up = nn.ModuleList([nn.Linear(time_emb_dim, 128),
                                          nn.Linear(time_emb_dim, 64),
                                          nn.Linear(time_emb_dim, 32),
                                          nn.Linear(time_emb_dim, 16),
                                          #nn.Linear(time_emb_dim, 3)
                                         ])
        
        
        if self.num_branches == 6:
            self.linear_up = nn.Linear(latent_size, 11776)
        else:
            self.linear_up = nn.Linear(latent_size, 35328)
            
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.len = len(self.up)
        

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        
        residual_input = []
        
        x = self.first_conv(x)
        
        # down
        for down in range(self.len):
            
            down_emb = self.relu(self.time_emb_down[down](t))
            down_emb = down_emb[(..., ) + (None, ) * 2]
            
            x = self.elu(self.down[down](x))

            x = x + down_emb

            residual_input.append(x)
        
        x = self.flatten(x)
        x = self.linear_down(x)
        
        # up
        x = self.elu(self.linear_up(x))
        x = x.view(-1, 256, 23, 6)
            
        for up in range(self.len):
            #print("\n")
            #print("up number:", up)
            residual_x = residual_input.pop()
            #print("x:", x.shape)
            #print("residual:", residual_x.shape)

            x = torch.cat((x, residual_x), dim = 1)

            #print("concat:", x.shape)
            
            up_emb = self.relu(self.time_emb_up[up](t))
            up_emb = up_emb[(..., ) + (None, ) * 2]

            #print("up emb:", up_emb.shape)
            
            x = self.elu(self.up[up](x))

            #print("up:", x.shape)
            
            x = x + up_emb

            #print("final:", x.shape)
            
        x = self.relu(self.last_conv(x))
        
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.T)
    
    def cosine_schedule(self, s=0.008):
        def f(t, s):
            return torch.cos((t / self.T + s) / (1 + s) * 0.5 * torch.pi) ** 2
        x = torch.linspace(0, self.T, self.T + 1)
        alphas_cumprod = f(x, s) / f(torch.tensor([0]), s)
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clip(betas, 0.0001, 0.999)
        return betas

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.T, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm.tqdm(reversed(range(1, self.T)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        self.time_dim = torch.tensor(time_dim).to(device)
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        ).to(device)
        pos_enc_a = torch.sin(t.repeat(1, channels.to(device) // 2) * inv_freq).to(device)
        pos_enc_b = torch.cos(t.repeat(1, channels.to(device) // 2) * inv_freq).to(device)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
            t = t.unsqueeze(-1).type(torch.float).to(device)
            t = self.pos_encoding(t, self.time_dim)

            x1 = self.inc(x)
            x2 = self.down1(x1, t)
            x2 = self.sa1(x2)
            x3 = self.down2(x2, t)
            x3 = self.sa2(x3)
            x4 = self.down3(x3, t)
            x4 = self.sa3(x4)

            x4 = self.bot1(x4)
            x4 = self.bot2(x4)
            x4 = self.bot3(x4)
            
            x = self.up1(x4, x3, t)
            x = self.sa4(x)
            x = self.up2(x, x2, t)
            x = self.sa5(x)
            x = self.up3(x, x1, t)
            x = self.sa6(x)
            output = self.outc(x)
            return output
    

class LatentDiffusionModel_mlp(nn.Module):
    def __init__(self, input_size = 300, hidden_size = 128, output_size = 300):
        super(LatentDiffusionModel_mlp, self).__init__()
        self.time_emb_dim = hidden_size

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
                nn.ReLU()
            )
        
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Define MLP layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size * 3)
        self.fc3 = nn.Linear(hidden_size * 3, hidden_size * 4)
        self.fc4 = nn.Linear(hidden_size * 4, hidden_size * 3)
        self.fc5 = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.fc6 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc7 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        x0 = self.relu(self.fc1(x))
        x1 = torch.cat([x0, time_emb], dim = 1)
        x2 = self.relu(self.fc2(x1))
        x3 = self.relu(self.fc3(x2))
        x4 = self.relu(self.fc4(x3)) + x2
        x5 = self.relu(self.fc5(x4)) + x1
        x6 = self.relu(self.fc6(x5)) + x0
        x = self.fc7(x6)
        return x