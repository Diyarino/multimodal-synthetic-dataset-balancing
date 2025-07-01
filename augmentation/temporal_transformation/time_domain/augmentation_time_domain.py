# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:58:15 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - time aligned averaging -- spawner, wDBA
    - slicing Le Guennec A, Malinowski S, Tavenard R. Data augmentation for time series classification using convolutional
neural networks.
    - time window warping
    - interpolation --> selecting some points
"""

#%% imports

import torch

#%% argumentations

class DynamicTimeWarping(torch.nn.Module):
    def __init__(self, probability: float = 1., window: int = 1, features: int = 0):
        """
        Calc dynamic time warping of the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        window : int, optional
            The window length to calc the average.
        features: tuple or list, optional
            Amount of randomly selected features.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.window = window
        self.features = features
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some trend.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                inp = inp
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, window={}, features={})'.format(
            self.probability, self.window, self.features)
    

class MovingAverage(torch.nn.Module):
    def __init__(self, probability: float = 1., window: int = 1, features: int = 0):
        """
        Calc moving average of the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        window : int, optional
            The window length to calc the average.
        features: tuple or list, optional
            Amount of randomly selected features.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.window = window
        self.features = features
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some trend.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                sums = [inp[idx : self.window + idx, feature].sum() for idx in range(inp.shape[0] - self.window)]
                inp[self.window:, feature] = torch.tensor(sums)/self.window
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, window={}, features={})'.format(
            self.probability, self.window, self.features)
    
class Scale(torch.nn.Module):
    def __init__(self, probability: float = 1., lower_value: float = 0., upper_value: float = 1., features: int = 0):
        """
        Scale the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        min, max : float, optional
            The min and max parameter of the linear function.
        features: tuple or list, optional
            Amount of randomly selected features.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.lower_value = lower_value
        self.upper_value = upper_value
        self.features = features
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some trend.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                inp[:, feature] = ((self.upper_value-self.lower_value)*(inp[:, feature] - inp[:, feature].min())/(inp[:, feature].max() - inp[:, feature].min()))+self.lower_value
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, lower_value={}, upper_value={}, features={})'.format(
            self.probability, self.lower_value, self.upper_value, self.features)

class Clamp(torch.nn.Module):
    def __init__(self, probability: float = 1., min_: float = 0., max_: float = 1., features: int = 0):
        """
        Clamp the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        min, max : float, optional
            The min and max parameter of the linear function.
        features: tuple or list, optional
            Amount of randomly selected features.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.min = min_
        self.max = max_
        self.features = features
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some trend.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                inp[:, feature] = inp[:, feature].clamp(self.min, self.max)
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, min={}, max={}, features={})'.format(
            self.probability, self.min, self.max, self.features)

class Trend(torch.nn.Module):
    def __init__(self, probability: float = 1., slope: float = 1., bias: float = 0., features: int = 0):
        """
        Add trend to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        slope, bias : float, optional
            The max parameter of the linear function.
        features: tuple or list, optional
            Amount of randomly selected features.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.max_slope = slope
        self.max_bias = bias
        self.features = features
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some trend.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                slope, bias = torch.rand(1)*self.max_slope, torch.rand(1)*self.max_bias
                linearity = slope * torch.arange(0, inp.shape[0]) + bias 
                inp[:, feature] += linearity
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, slope={}, bias={}, features={})'.format(
            self.probability, self.slope, self.bias, self.features)

class Event(torch.nn.Module):
    def __init__(self, probability: float = 1., gain: float = 1., max_sig: float = 10., features: int = 0):
        """
        Add gaussian events to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        gain : float, optional
            The gain of the events.
        max_sig : float, optional
            Defines the maximum value of the standard deviation.
        features: tuple or list, optional
            Amount of randomly selected features.
        
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.gain = gain
        self.features = features
        self.max_sig = max_sig
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some events.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                x = torch.arange(0, inp.shape[0])
                mu, sig = torch.rand(1)*inp.shape[0], torch.rand(1)*self.max_sig
                peak = self.gain * torch.rand(1) * torch.exp(-(x - mu)**2 / (2 * (sig**2)))
                inp[:, feature] += peak
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, gain={}, max_sig={}, features={})'.format(
            self.probability, self.gain, self.max_sig, self.features)

class Seasonality(torch.nn.Module):
    def __init__(self, probability: float = 1., gain: float = 1., frequency: float = 0.01, features: int = 0):
        """
        Add seasonality to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        gain : float, optional
            The max gain of the events.
        frequency : float, optional
            Defines the maximum value of the frequency.
        features: tuple or list, optional
            Amount of randomly selected features.
        
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.gain = gain
        self.frequency = frequency
        self.features = features
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some events.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                frequency, gain = torch.rand(1)*self.frequency, torch.rand(1)*self.gain
                x = torch.arange(0, 1, 1/inp.shape[0]) * 2 * 3.14 * frequency
                season = gain * torch.sin(x)
                inp[:, feature] += season
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, gain={}, frequency={}, features={})'.format(
            self.probability, self.gain, self.frequency, self.features)

class Step(torch.nn.Module):
    def __init__(self, probability: float = 1., gain: float = 1., features: int = 0):
        """
        Add step event to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        gain : float, optional
            The max gain of the events.
        features: tuple or list, optional
            Amount of randomly selected features.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.gain = gain
        self.features = features
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some step events.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                start = torch.randint(0, inp.shape[0], (1,))
                step = torch.zeros(inp.shape[0])
                step[start:] += 1
                inp[:, feature] += self.gain * torch.rand(1) * step
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, gain={}, features={})'.format(
            self.probability, self.gain, self.features)

class Exp(torch.nn.Module):
    def __init__(self, probability: float = 1., gain: float = 1., features: int = 0):
        """
        Add exp event to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        gain : float, optional
            The max gain of the events. Maximum value should be 5.
        features: tuple or list, optional
            Amount of randomly selected features.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.gain = gain
        self.features = features
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some exp events.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                x = torch.arange(0, 1, 1/inp.shape[0])
                exp = torch.exp(self.gain * torch.rand(1) * x)
                inp[:, feature] += exp
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, gain={}, features={})'.format(
            self.probability, self.gain, self.features)
    
class Log(torch.nn.Module):
    def __init__(self, probability: float = 1., gain: float = 1., features: int = 0):
        """
        Add log event to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        gain : float, optional
            The max gain of the events. Maximum value should be 5.
        features: tuple or list, optional
            Amount of randomly selected features.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.gain = gain
        self.features = features
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some log events.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                x = torch.arange(0, 1, 1/inp.shape[0])
                log = torch.log(self.gain * torch.rand(1) * x)
                inp[:, feature] += log
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, gain={}, features={})'.format(
            self.probability, self.gain, self.features)

class Flipping(torch.nn.Module):
    def __init__(self, probability: float = 1., dims: tuple = (1, 0)):
        """
        Flipps some of the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        dims : tuple, optional
            The dimension where flipping is applied. The default is (0, 1).
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.dims = dims
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs to be flipped.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            inp = inp.flip(self.dims)
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, dims={})'.format(
            self.probability, self.dims)

class Negation(torch.nn.Module):
    def __init__(self, probability: float = 1.):
        """
        Add negation to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        
        Returns
        -------
        None.
    
        """
        super().__init__()
        self.probability = probability
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some negation.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if self.probability>torch.rand(1):
            inp=-inp
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={})'.format(self.probability)

class AddOffset(torch.nn.Module):
    def __init__(self, probability: float = 1., low: float = -1., high: float = 1.):
        """
        Add some offset to the elements of the input tensor. 
        
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        low, high : float, optional
            limitates the offset in a range.
            
        Returns
        -------
        None.
    
        """
        super().__init__()
        self.probability = probability
        self.low = low
        self.high = high
    
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some offsets.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if self.probability>torch.rand(1):
            # print("offset")
            inp = inp+torch.rand(1)*(self.high-self.low)+self.low
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, low={}, high={})'.format(
            self.probability, self.low, self.high)
    
class Gausnoise(torch.nn.Module):
    def __init__(self, probability: float = 1., mean: float = 0, std: float = 1.):
        """
        Add gaussian noise to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        mean, p_std : float, optional
            Parameters of the gaussian noise.
            
        Returns
        -------
        None.
    
        """
        super().__init__()
        self.probability = probability
        self.mean = mean
        self.std = std

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some offsets.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if self.probability>torch.rand(1):
            # print("gausinannoise")
            inp = inp + torch.randn(inp.size()) * self.std + self.mean
        return inp    
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, mean={}, std={})'.format(
            self.probability, self.mean, self.std)
    
class PermuteFeatures(torch.nn.Module):
    def __init__(self, probability: float = 1.):
        """
        Permutes features of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
            
        Returns
        -------
        None.
    
        """
        super().__init__()
        self.probability = probability

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some offsets.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if self.probability>torch.rand(1):
            # print("permute channels")
            permute = torch.randperm(inp.shape[1])
            inp = inp[:, permute]
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={})'.format(self.probability)
    
class Magnitude(torch.nn.Module):
    def __init__(self, probability: float = 1., negative: float = 0.5):
        """
        Add gain to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        negative : float, optional
            Add negation to the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        
        Returns
        -------
        None.
    
        """
        super().__init__()
        self.probability = probability
        self.negative = negative
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some offsets.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if self.probability>torch.rand(1):
            # print("absolut inp")
            inp = inp.abs()
            if self.negative>torch.rand(1):
               inp=-inp
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, negative={})'.format(
            self.probability, self.negative)
    
    
class Permutation(torch.nn.Module):
    def __init__(self, probability: float = 1., pieces: int = 3, features: int = 1):
        """
        Split elements of the input tensor in pieces. 
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        pieces : int, optional
            Amount of pieces in rows and colums. The default is 3.
            
        Returns
        -------
        None.
            
        """
        super().__init__()
        self.probability = probability
        self.pieces = pieces
        self.features = features

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs to be puzzled. Have to be 3 dimensional.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if self.probability>torch.rand(1):
            x_range = int(inp.shape[0] / self.pieces)
            features = torch.randint(0, inp.shape[1], (self.features,))
            for feature in features:
                buffer = torch.zeros(self.pieces, x_range)
                for piece in range(self.pieces):
                    buffer[piece, :] = inp[piece*x_range:piece*x_range+x_range, feature]
                buffer = buffer[torch.randperm(buffer.shape[0])]
                inp[:, feature] = buffer.reshape(self.pieces*buffer.shape[1])
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, pieces={}, features={})'.format(
            self.probability, self.pieces, self.features)
    
class Replacer(torch.nn.Module):
    def __init__(self, probability: float = 1., size: tuple = (0,0), replace_val: float = 0.):
        """
        Randomly erase elements of the input tensor in rectangular form. 
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        size : tuple, optional
            Size of the rectangle in rows and colums. The default is (0,0).
        replace_val : float, optional
            The value of the elements in the rectangle. The default value 0. paint it black.
            
        Returns
        -------
        None.
            
        """
        super().__init__()
        self.probability = probability
        self.size = size
        self.replace_val = replace_val
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs to be padded.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if self.probability>torch.rand(1):
            rows, cols = inp.size()
            features = torch.randint(0, cols, (self.size[1],))
            for idx in features:
                rand_pos_y = torch.randint(0, rows-self.size[0], (1,))
                rand_len = torch.randint(0, self.size[0], (1,))
                inp[rand_pos_y : rand_pos_y + rand_len, idx] = self.replace_val
        return inp

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, size={}, replace_val={})'.format(
            self.probability, self.size, self.replace_val)
    

