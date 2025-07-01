# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:16:17 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - filters
    - convolution
    - dynamic frequency warping
    - SFM
    - handle with complex data at output
"""

# %% imports

import torch

# %% argumentations


class Fourier(torch.nn.Module):
    def __init__(self, probability: float = 1.):
        """
        Apply fourier transformation to the input tensor. 

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
            The input array which needs to be padded.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            inp = torch.fft.fft(inp)
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


class IFourier(torch.nn.Module):
    def __init__(self, probability: float = 1.):
        """
        Apply inverse fourier transformation to the input tensor. 

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
            The input array which needs to be padded.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            inp = torch.fft.ifft(inp)
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


class FreqShift(torch.nn.Module):
    def __init__(self, probability: float = 1., shift: int = 1, features: int = 0):
        """
        Apply shift to fourier transformation to the input tensor. 

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
        self.shift = shift
        self.features = features

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
        if self.probability > torch.rand(1):
            transfer = torch.fft.fft(inp)

            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                transfer[:, feature] = transfer[:, feature].roll(torch.randint(0, self.shift, (1,)).item())

            phase = torch.angle(transfer)
            magnitude = torch.abs(transfer)
            transfer = torch.polar(magnitude, phase)
            inp = transfer
            # inp = torch.fft.ifft(transfer)
        return inp

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, shift={}, features={})'.format(
            self.probability, self.shift, self.features)


class FreqMasking(torch.nn.Module):
    def __init__(self, probability: float = 1.):
        """
        Randomly erase elements of the input tensor in rectangular form. 

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
            The input array which needs to be padded.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            transfer = torch.fft.fft(inp)
            phase = torch.angle(transfer)
            magnitude = torch.abs(transfer)
            transfer = torch.polar(magnitude, phase)
            inp = torch.fft.ifft(transfer)
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


class PhaseManipulation(torch.nn.Module):
    def __init__(self, probability: float = 1., gain: float = 0., features: int = 0):
        """
        Randomly erase elements of the input tensor in rectangular form. 

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
        self.gain = gain
        self.features = features

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
        if self.probability > torch.rand(1):
            transfer = torch.fft.fft(inp)
            phase = torch.angle(transfer)
            magnitude = torch.abs(transfer)

            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                phase[:, feature] = self.gain * torch.rand(1) * phase[:, feature]

            transfer = torch.polar(magnitude, phase)
            inp = torch.fft.ifft(transfer)
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


class PhaseScalling(torch.nn.Module):
    def __init__(self, probability: float = 1., lower_value: float = 0., upper_value: float = 1., features: int = 0):
        """
        Randomly scale magnitude. 

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
        self.lower_value = lower_value
        self.upper_value = upper_value
        self.features = features

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
        if self.probability > torch.rand(1):
            transfer = torch.fft.fft(inp)
            phase = torch.angle(transfer)
            magnitude = torch.abs(transfer)

            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                phase[:, feature] = ((self.upper_value-self.lower_value) *
                                     (phase[:, feature] - phase[:, feature].min()) /
                                     (phase[:, feature].max() - phase[:, feature].min())
                                     )+self.lower_value

            transfer = torch.polar(magnitude, phase)
            inp = torch.fft.ifft(transfer)
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


class MagnitudeManipulation(torch.nn.Module):
    def __init__(self, probability: float = 1., gain: float = 1.0,  features: int = 0):
        """
        Randomly add gain. 

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
        self.gain = gain
        self.features = features

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
        if self.probability > torch.rand(1):
            transfer = torch.fft.fft(inp)
            phase = torch.angle(transfer)
            magnitude = torch.abs(transfer)

            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                magnitude[:, feature] = self.gain * torch.rand(1) * magnitude[:, feature]

            transfer = torch.polar(magnitude, phase)
            inp = torch.fft.ifft(transfer)
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


class MagnitudeScalling(torch.nn.Module):
    def __init__(self, probability: float = 1., lower_value: float = 0., upper_value: float = 1., features: int = 0):
        """
        Randomly scale magnitude. 

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
        self.lower_value = lower_value
        self.upper_value = upper_value
        self.features = features

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
        if self.probability > torch.rand(1):
            transfer = torch.fft.fft(inp)
            phase = torch.angle(transfer)
            magnitude = torch.abs(transfer)

            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                magnitude[:, feature] = ((self.upper_value-self.lower_value) *
                                         (magnitude[:, feature] - magnitude[:, feature].min()) /
                                         (magnitude[:, feature].max() - magnitude[:, feature].min())
                                         )+self.lower_value

            transfer = torch.polar(magnitude, phase)
            inp = torch.fft.ifft(transfer)
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
