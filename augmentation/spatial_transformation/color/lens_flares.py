# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    
    # Berechne die fusionierten RGB-Werte mit der Alpha-Kanal-Mischformel
    # rgb_fusion = (alpha1 * rgb1 + alpha2 * rgb2 * (1 - alpha1)) / (alpha1 + alpha2 * (1 - alpha1))
    
    # # Berechne den fusionierten Alpha-Kanal mit der Alpha-Kanal-Mischformel
    # alpha_fusion = alpha1 + alpha2 * (1 - alpha1)
    
    # # Füge die fusionierten RGB- und Alpha-Kanalwerte wieder zu einem Pixel zusammen
    # pixel_fusion = torch.cat([rgb_fusion, alpha_fusion], dim=0)

"""

# %% imports

import torch
from torchvision.transforms import GaussianBlur

# %% dirt stains

class Flares(torch.nn.Module):
    def __init__(self, probability: float = .5, num_flares: int = 100, radius: tuple = (1, 15), 
                 mode: str = 'circ', kernel_size: tuple = (25, 25), sigma: tuple = (15, 20)):
        '''
        This class add some stains to the image that is looks like a dirty lense.
        Btw. without the alpha channel.

        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        num_flares : int, optional
            Amount of dirt stains. A random value from zero to this value is choosen. The default is 100.
        radius : tuple, optional
            The min and max radius of the particles. The default is (1, 15).
        mode : str, optional
            Mode if rect or circles as dirt should be choosen. The default is 'rect'.
        kernel_size : int or sequence, optional
            Size of the Gaussian kernel. The default is (25 ,25).
        sigma : float or tuple of float (min, max), optional
            Standard deviation to be used for creating kernel to perform blurring. If float, sigma is fixed. 
            If it is tuple of float (min, max), sigma is chosen uniformly at random to lie in the given range.
            The default is (15, 20).

        Returns
        -------
        None.

        '''
        super().__init__()
        self.probability = probability
        self.num_flares = num_flares
        self.radius = radius
        self.min_radius = radius[0]
        self.max_radius = radius[1] + 1
        self.mode = mode
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        
    def __repr__(self):
        """
        Represent the class.
    
        Returns
        -------
        str
            The representation of the class.
    
        """
        return self.__class__.__name__+'(probability={}, num_flares={}, radius={}, mode={})'.format(
            self.probability, self.num_flares, self.radius, self.mode)
    

    def forward(self, inp: torch.Tensor):
        """
        Call argumentation.
        
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some stains.
        
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
        
        """
        if torch.rand(1) <= self.probability:
            device = inp.device
            inp = inp.clone().to(device)
            clamp_params = (0.,1.) if 'Float' in inp.type() else (0,255)
            *batch, rgba, cols, rows = inp.shape
            num_pieces = torch.randint(1, self.num_flares, (1,))
            x_centers = torch.randint(0, rows, (num_pieces,))
            y_centers = torch.randint(0, cols, (num_pieces,))
            radii = torch.randint(self.min_radius, self.max_radius, (num_pieces,))
            colors = torch.rand(num_pieces, 3, device=device)*inp.max()
            flares_tensor = torch.zeros_like(inp, device = device)

            if self.mode == 'rect':
                for x,y,flare_radius,flare_color in zip(x_centers, y_centers, radii, colors):
                    draw_tensor = torch.zeros_like(flares_tensor)
                    flare_color = flare_color.unsqueeze(-1).unsqueeze(-1)
                    draw_tensor[..., y-flare_radius:y+flare_radius, x-flare_radius:x+flare_radius] = flare_color
                    flares_tensor += draw_tensor
                
            if self.mode == 'circ':
                Y, X = torch.arange(cols).unsqueeze(0).permute(1, 0), torch.arange(rows).unsqueeze(0)
                for x,y,flare_radius,flare_color in zip(x_centers, y_centers, radii, colors):
                    dist_from_center = torch.sqrt((X - x)**2 + (Y-y)**2).to(device)
                    flare_color = flare_color.unsqueeze(-1).unsqueeze(-1).to(device)
                    draw_tensor = (dist_from_center <= flare_radius).expand_as(inp)*flare_color
                    flares_tensor += draw_tensor.type(inp.type())           
                
            # Apply a blur filter to the flares using PyTorch operations
            flares_blur_tensor = self.blur(flares_tensor)
            
            inp = (inp + 0.99*flares_blur_tensor).clamp(*clamp_params).type(inp.type()) 

        return inp
    
    
    
        

# %% test

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.transforms import PILToTensor
    
    canvas = PILToTensor()(Image.open('D:/Sciebo/04_GitLab/augmentation_techniques/resources/001503.png'))
    augmentation = Flares(probability=1, num_flares = 10, radius = (1, 25), mode = 'circ')
    new = augmentation(canvas).permute(1,2,0)
    plt.imshow(new)
