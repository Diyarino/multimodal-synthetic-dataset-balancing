# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - type stuff
"""

# %% imports

import torch

# %% bresenham algorthm

def bresenham(x1, y1, x2, y2):
    """Zeichnet eine Linie von (x1, y1) nach (x2, y2) im Raster.

    Args:
        x1 (int): x-Koordinate des Startpunkts.
        y1 (int): y-Koordinate des Startpunkts.
        x2 (int): x-Koordinate des Endpunkts.
        y2 (int): y-Koordinate des Endpunkts.

    Returns:
        List[Tuple[int, int]]: Eine Liste von (x, y)-Koordinaten, die die Linie repräsentieren.
    """
    line = []
    dx, dy = abs(x2 - x1), abs(y2 - y1)

    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy

    while x1 != x2 or y1 != y2:
        line.append((x1, y1))
        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            x1 += sx
        if err2 < dx:
            err += dx
            y1 += sy

    line.append((x2, y2))  # Füge den Endpunkt hinzu

    return line

# %% scratches


class Scatches(torch.nn.Module):
    def __init__(self, probability: float = 0.5, num_cracks: int= 100, max_length: int = 2, max_width: int = 2):
        '''
        This class adds some scratches to the image.

        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        num_cracks, max_length, max_width : int, optional
            Defines the amount, the max length an width of the scratches..

        Returns
        -------
        None.

        '''
        super().__init__()
        self.probability = probability
        self.num_cracks = num_cracks
        self.max_length = max_length
        self.max_width = max_width
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, num_cracks={}, max_length={}, max_width={})'.format(
            self.probability, self.num_cracks, self.max_length, self.max_width)
        
    def forward(self, inp: torch.Tensor, batch_first = False):
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some scratches.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if torch.rand(1) <= self.probability:
            inp = inp.clone()
            val = inp.max()
            if batch_first:
                batch, rgb, cols, rows = inp.shape
            else:
                rgb, cols, rows = inp.shape
            
            num_cracks = torch.randint(1, self.num_cracks,(1,)) # Number of cracks
            # Randomly set the start and end points of the crack
            x_start, x_end = torch.randint(0,rows,(2,num_cracks)) 
            y_start, y_end = torch.randint(0,cols,(2,num_cracks))
            # Randomly set the length and width of the crack
            length = torch.randint(1, self.max_length,(num_cracks,))
            width = torch.randint(1, self.max_width,(num_cracks,))
            
            dx, dy = abs(x_end - x_start), abs(y_end - y_start)
            sx = (x_start < x_end).long()*2-1
            sy = (y_start < y_end).long()*2-1
            err = dx - dy
            
            data = zip(x_start, x_end, y_start,y_end, length, width, dx, dy, sx, sy, err)
            for i, (x_start, x_end, y_start,y_end, length, width, dx, dy, sx, sy, err) in enumerate(data):
                while x_start != x_end or y_start != y_end:
                    inp[...,y_start:y_start+width, x_start:x_start+length] = val
                    
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x_start += sx
                    if e2 < dx:
                        err += dx
                        y_start += sy
                    
        return inp


# %% test

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    canvas = torch.zeros(3,256,256).int()
    canvas[1,1,1] +=255
    transform = Scatches()
    
    plt.imshow(transform(canvas).permute(1,2,0), cmap='gray')