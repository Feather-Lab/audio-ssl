import torch
import torchaudio
import random
import numpy as np
import scipy
import sys
import chcochleagram
from chcochleagram import compression
from chcochleagram import cochleagram
from chcochleagram import *
# from torchaudio.sox_effects import apply_effects_tensor
import sox 


def ch_demean(x, dim=0):
    '''
    Helper function to mean-subtract tensor.
    
    Args
    ----
    x (tensor): tensor to be mean-subtracted
    dim (int): kwarg for torch.mean (dim along which to compute mean)
    
    Returns
    -------
    x_demean (tensor): mean-subtracted tensor
    '''
    x_demean = torch.sub(x, torch.mean(x, dim=dim))
    return x_demean


def ch_rms(x, dim=0):
    '''
    Helper function to compute RMS amplitude of a tensor.
    
    Args
    ----
    x (tensor): tensor for which RMS amplitude should be computed
    dim (int): kwarg for torch.mean (dim along which to compute mean)
    
    Returns
    -------
    rms_x (tensor): root-mean-square amplitude of x
    '''
    rms_x = torch.sqrt(torch.mean(torch.pow(x, 2), dim=dim))
    return rms_x

def np_demean(x, axis=0):
    '''
    Helper function to mean-subtract tensor.
    
    Args
    ----
    x (nd.array): array to be mean-subtracted
    axis (int): kwarg for numpy.mean (axis along which to compute mean)
    
    Returns
    -------
    x_demean (nd.array): mean-subtracted tensor
    '''
    x_demean = np.subtract(x, np.mean(x, axis=axis))
    return x_demean


def np_rms(x, axis=0):
    '''
    Helper function to compute RMS amplitude of a tensor.
    
    Args
    ----
    x (np.ndarray): tensor for which RMS amplitude should be computed
    axis (int): kwarg for np.mean (axis along which to compute mean)
    
    Returns
    -------
    rms_x (np.ndarray): root-mean-square amplitude of x
    '''
    rms_x = np.sqrt(np.mean(np.power(x, 2), axis=axis))
    return rms_x


class AudioCompose(torch.nn.Module):
    """
    Composes several foreground/background audio transforms together (based off of 
        torchvision.transforms.Compose)

    Args:
        transforms (list of audio_function transfrom torch.nn.Modules): list of transforms to compose. 

    """

    def __init__(self, transforms):
        super(AudioCompose, self).__init__()
        self.transforms = transforms

    def forward(self, foreground_wav, background_wav):
        for t in self.transforms:
            foreground_wav, background_wav = t(foreground_wav, background_wav)
        return foreground_wav, background_wav

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class LogScaleFakeClipping(torch.nn.Module):
    """
    Scales the values by a log scale. (Useful to apply aftr the Mel Spectrogram)
    """
    def __init__(self, offset=1e-6):
        super(LogScaleFakeClipping, self).__init__()
        self.offset = offset
        self.clamp_function = FakeClamp.apply 

    def forward(self, foreground_wav, background_wav):
        foreground_wav = self.clamp_function(foreground_wav, self.offset)
        foreground_wav = torch.log2(foreground_wav)
        if background_wav is not None:
            background_wav = self.clamp_function(background_wav, self.offset)
            background_wav = torch.log2(background_wav)
        return foreground_wav, background_wav

class FakeClamp(torch.autograd.Function):
    """
    Applies clamp in the forward pass, but all gradients=1 in the backwards
    pass.
    """
    @staticmethod
    def forward(ctx, x, min):
        return torch.clamp(x, min=min)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class LogScale(torch.nn.Module):
    """
    Scales the values by a log scale. (Useful to apply aftr the Mel Spectrogram)
    """
    def __init__(self, offset=1e-6):
        super(LogScale, self).__init__()
        self.offset = offset

    def forward(self, foreground_wav, background_wav):        
        foreground_wav = torch.clamp(foreground_wav, min=self.offset)
        foreground_wav = torch.log2(foreground_wav)
        if background_wav is not None:
            background_wav = torch.clamp(background_wav, min=self.offset)
            background_wav = torch.log2(background_wav)
        return foreground_wav, background_wav

class ClippedGradPower(torch.nn.Module):
    """
    Wrapper around ClippedGradPowerCompression defined in chcochleagram.compression
    """
    def __init__(self, compression_kwargs):
        super(ClippedGradPower, self).__init__()
        self.compression_kwargs = compression_kwargs
        self.compression_function = compression.ClippedGradPowerCompression(**compression_kwargs)

    def forward(self, foreground_wav, background_wav):
        foreground_wav = self.compression_function(foreground_wav)
        if background_wav is not None:
            background_wav = self.compression_function(background_wav)
        return foreground_wav, background_wav


class AudioToAudioRepresentation(torch.nn.Module):
    """
    Base class for audio transformations. Takes in the audio and outputs
    a representation that is used for training. 
    Args:
        rep_type (str): the type of representation to build
    """
    def __init__(self, rep_type, rep_kwargs, compression_type, compression_kwargs):
        super(AudioToAudioRepresentation, self).__init__()
        self.rep_type = rep_type
        self.rep_kwargs = rep_kwargs
        self.compression_type = compression_type
        self.compression_kwargs = compression_kwargs

        # Choose the representation type
        if self.rep_type == 'mel_spec':
            self.rep = AudioToMelSpectrogram(melspec_kwargs=self.rep_kwargs)
        elif self.rep_type == 'cochleagram':
            self.rep = AudioToCochleagram(cgram_kwargs=self.rep_kwargs)
        else:
            raise NotImplementedError('Audio Representation of type '
              '%s is not implemented'%self.rep_type)

        # Choose the compression type
        if self.compression_type == 'log':
            self.compression = LogScale(**self.compression_kwargs)
        elif self.compression_type == 'log_fakeclamp':
            self.compression = LogScaleFakeClipping(**self.compression_kwargs)
        elif self.compression_type == 'coch_p3':
            self.compression = ClippedGradPower(self.compression_kwargs)
        elif self.compression_type == 'none':
            self.compression = None
        else:
            raise NotImplementedError('Audio Compression of type '
               '%s is not implemented'%self.compression_type)
    
    def forward(self, foreground_wav, background_wav):
        del background_wav
        if foreground_wav is not None:
            foreground_wav = foreground_wav
            foreground_rep, background_rep = self.rep(foreground_wav, None)
            if self.compression is not None:
                foreground_rep, background_rep = self.compression(foreground_rep, None)
        else:
            foreground_rep = None
            background_rep = None
        return foreground_rep, background_rep


class AudioToMelSpectrogram(torch.nn.Module):
    """
    Converts audio to mel spectrogram. 
    Args: 
        melspec_kwargs (dict): dictionary containing the arguments used within
            torchaudio.MelSpectrogram
    """
    def __init__(self, melspec_kwargs={}):
        super(AudioToMelSpectrogram, self).__init__()
        self.melspec_kwargs = melspec_kwargs
        self.MelSpectrogram = torchaudio.transforms.MelSpectrogram(**self.melspec_kwargs)
    
    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        del background_wav

        if foreground_wav is not None:
            foreground_mel = self.MelSpectrogram(foreground_wav)
        else:
            foreground_mel = None

        return foreground_mel, None


class AudioToCochleagram(torch.nn.Module):
    """
    Converts audio to cochleagram
    """
    def __init__(self, cgram_kwargs={}):
        super(AudioToCochleagram, self).__init__()
        self.cgram_kwargs = cgram_kwargs

        # Args used for multiple of the cochleagram operations
        self.signal_size = self.cgram_kwargs['signal_size']
        self.sr = self.cgram_kwargs['sr']
        self.pad_factor = self.cgram_kwargs['pad_factor']
        self.use_rfft = self.cgram_kwargs['use_rfft']

        # Define cochlear filters
        self.coch_filter_kwargs = self.cgram_kwargs['coch_filter_kwargs']
        self.coch_filter_kwargs = {'use_rfft':self.use_rfft,
                                   'pad_factor':self.pad_factor,
                                   'filter_kwargs':self.coch_filter_kwargs}
 
        self.make_coch_filters = self.cgram_kwargs['coch_filter_type']
        self.filters = self.make_coch_filters(self.signal_size,
                                              self.sr, 
                                              **self.coch_filter_kwargs)

        # Define an envelope extraction operation
        self.env_extraction = self.cgram_kwargs['env_extraction_type']
        self.envelope_extraction = self.env_extraction(self.signal_size, 
                                                       self.sr, 
                                                       self.use_rfft, 
                                                       self.pad_factor)

        # Define a downsampling operation
        self.downsampling = self.cgram_kwargs['downsampling_type']
        self.env_sr = self.cgram_kwargs['env_sr']
        self.downsampling_kwargs = self.cgram_kwargs['downsampling_kwargs']
        self.downsampling_op = self.downsampling(self.sr, self.env_sr, **self.downsampling_kwargs)

        # Compression is applied as a separate transform to be consistent with Spectrograms
        cochleagram = chcochleagram.cochleagram.Cochleagram(self.filters, 
                                                            self.envelope_extraction,
                                                            self.downsampling_op,
                                                            compression=None)

        self.Cochleagram = cochleagram

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        del background_wav

        if foreground_wav is not None:
            foreground_coch = self.Cochleagram(foreground_wav)
        else:
            foreground_coch = None

        return foreground_coch, None
        

class AudioToTensor(torch.nn.Module):
    """
    Convert the foreground and background sounds to tensors

    Args:
        None

    Returns:
        foreground_wav, background_wav
    """
    def __init__(self):
        super(AudioToTensor, self).__init__()

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if background_wav is None:
            return torch.from_numpy(foreground_wav), None
        else:
            return torch.from_numpy(foreground_wav), torch.from_numpy(background_wav)


class UnsqueezeAudio(torch.nn.Module):
    """
    Adds a channel dimension (useful for mel-spectrograms as inputs)

    Args:
        None

    Returns:
        foreground_wav, background_wav
    """
    def __init__(self, dim=1):
        super(UnsqueezeAudio, self).__init__()
        self.dim = dim

    def forward(self, foreground_wav, background_wav):
        if foreground_wav is not None:
            foreground_wav = foreground_wav.unsqueeze(self.dim)
        if background_wav is not None:
            background_wav = background_wav.unsqueeze(self.dim)
        return foreground_wav, background_wav


class FilterNoneSpeech(torch.nn.Module):
    """
    Filter out speech audio samples that are all zeros. 
    Useful for removing speech 'null' classes. 

    Args:
        None

    Returns:
        foreground_wav, background_wav if passes filtering
        None if should be removed
    """
    def __init__(self):
        super(FilterNoneSpeech, self).__init__()

    def forward(self, foreground_wav, background_wav): 
        if torch.sum(torch.pow(foreground_wav, 2))==0:
            foreground_wav = None
        if torch.sum(torch.pow(background_wav, 2))==0:
            background_wav = None
        else:
            return foreground_wav, background_wav


class RandomCropForegroundBackground(torch.nn.Module):
    """
    Randomly crops the foreground and background to make a shorter signal. 
    """
    def __init__(self, signal_size, crop_length):
        super(RandomCropForegroundBackground, self).__init__()
        self.crop_length = crop_length
        self.signal_size = signal_size
        self.start_crop = int(signal_size - crop_length)

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        rand_start = torch.randint(self.start_crop, size=(2,))
        if foreground_wav is not None:
            foreground_wav = foreground_wav[rand_start[0]:rand_start[0]+self.crop_length]
        if background_wav is not None:
            background_wav = background_wav[rand_start[1]:rand_start[1]+self.crop_length]
        return foreground_wav, background_wav

class CenterCropForegroundBackground(torch.nn.Module):
    """
    Center crops the foreground and background to make a shorter signal.
    """
    def __init__(self, signal_size, crop_length):
        super(CenterCropForegroundBackground, self).__init__()
        self.crop_length = crop_length
        self.signal_size = signal_size
        self.start_crop_center = int((signal_size-crop_length)/2)
        
    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if foreground_wav is not None:
            foreground_wav = foreground_wav[self.start_crop_center:self.start_crop_center+self.crop_length]
        if background_wav is not None:
            background_wav = background_wav[self.start_crop_center:self.start_crop_center+self.crop_length]
        return foreground_wav, background_wav
    
class CenterCropForegroundRandomCropBackground(torch.nn.Module):
    """
    Center crops the foreground and randomly crops background to make a shorter signal.
    """
    def __init__(self, signal_size, crop_length):
        super(CenterCropForegroundRandomCropBackground, self).__init__()
        self.crop_length = crop_length
        self.signal_size = signal_size
        self.start_crop_random = int(signal_size - crop_length)
        self.start_crop_center = int((signal_size-crop_length)/2)
        
    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        rand_start = torch.randint(self.start_crop_random, size=(2,))
        if foreground_wav is not None:
            foreground_wav = foreground_wav[self.start_crop_center:self.start_crop_center+self.crop_length]
        if background_wav is not None:
            background_wav = background_wav[rand_start[1]:rand_start[1]+self.crop_length]
        return foreground_wav, background_wav


class RMSNormalizeForegroundAndBackground(torch.nn.Module):
    """
    RMS normalize the foreground and background sounds

    Args:
        rms_normalization (float): The rms level to set the sound to

    Returns:
        foreground_wav, background_wav
    """
    def __init__(self, rms_level=0.1):
        super(RMSNormalizeForegroundAndBackground, self).__init__()
        self.rms_level=rms_level

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if foreground_wav is not None:
            foreground_wav = ch_demean(foreground_wav)
            rms_foreground = ch_rms(foreground_wav)
            if rms_foreground !=0:
                foreground_wav = foreground_wav * self.rms_level / rms_foreground
            else:
                foreground_wav = None

        if background_wav is not None:
            background_wav = ch_demean(background_wav)
            rms_background = ch_rms(background_wav)
            if rms_background !=0:
                background_wav = background_wav * self.rms_level / rms_background
            else:
                background_wav = None

        return foreground_wav, background_wav


class DBSPLNormalizeForegroundAndBackground(torch.nn.Module):
    """
    Set the foreground and background sounds to a specified sound pressure 
    level (dBSPL)

    Args:
        dbspl (float): desired sound pressure level in dB re 20e-6 Pa
        use_np (bool): Use torch or numpy operations

    Returns:
        foreground_wav, background_wav
    """
    def __init__(self, dbspl=60, use_np=False):
        super(DBSPLNormalizeForegroundAndBackground, self).__init__()
        self.dbspl=dbspl
        self.use_np=use_np
        self.rms_level = 20e-6 * np.power(10.0, self.dbspl / 20.0)
        if self.use_np:
            self.demean = np_demean
            self.rms = np_rms
        else:
            self.demean = ch_demean
            self.rms = ch_rms

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        if foreground_wav is not None:

            foreground_wav = self.demean(foreground_wav)
            rms_foreground = self.rms(foreground_wav)
            if rms_foreground !=0:
                foreground_wav = foreground_wav * self.rms_level / rms_foreground
            else:
                foreground_wav = None

        if background_wav is not None:
            background_wav = self.demean(background_wav)
            rms_background = self.rms(background_wav)
            if rms_background !=0:
                background_wav = background_wav * self.rms_level / rms_background
            else:
                background_wav = None

        return foreground_wav, background_wav


class FlipForegroundAndBackground(torch.nn.Module):
    """
    Turns the foreground signal into the background signal and 
    vice versa (useful for training without any combinations)

    Returns:
        foreground_wav, background_wav
    """
    def __init__(self):
        super(FlipForegroundAndBackground, self).__init__()

    def forward(self, foreground_wav, background_wav):
        """
        Args:
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as
                the background audio sample
        """
        return background_wav, foreground_wav 


class CombineWithRandomDBSNR(torch.nn.Module):
    """
    Takes two signals and combines them at a specified dB SNR level.
    
    Args: 
        low_snr (float): the low end for the range of dB SNR to draw from
        high_snr (float): the high end for the range of db SNR to draw from
        rms_level (float): the end RMS value for the combined sound

    Returns:
        signal_in_noise, None 

    """
    def __init__(self, low_snr=-10, high_snr=10):
        self.low_snr=low_snr
        self.high_snr=high_snr
        super(CombineWithRandomDBSNR, self).__init__()

    def forward(self, foreground_wav, background_wav):
        """
        Args:  
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as 
                the background audio sample
        """
        rand_db_snr = self.low_snr + (self.high_snr - self.low_snr) * torch.rand(1)
        rms_ratio = np.power(10.0, rand_db_snr / 20.0)
        # Demean signal and noise before computing rms
        if foreground_wav is not None:
            foreground_wav = ch_demean(foreground_wav)
            rms_foreground = ch_rms(foreground_wav)
        else:
            rms_foreground = 0
            foreground_wav = torch.zeros(background_wav.shape)
        if background_wav is not None:
            background_wav = ch_demean(background_wav)
            rms_background = ch_rms(background_wav)
        else:
            rms_background = 0
            background_wav = torch.zeros(foreground_wav.shape)

        # Calculate the scale factor for the two sounds
        # For now, to align with the jsinv3 dataset, we include the infinite SNR 
        # cases
        if rms_foreground == 0: # No foreground condition (just noise)
            noise_scale_factor = 1
        elif rms_background == 0: 
            noise_scale_factor = 0
        else:
            noise_scale_factor = torch.div(rms_foreground, 
                                           torch.mul(rms_background,
                                                     rms_ratio))
 
        background_wav = torch.mul(noise_scale_factor, background_wav)
        signal_in_noise = torch.add(foreground_wav, background_wav)

        return signal_in_noise, None

### New transforms for matched data ###
class RandomCrop:
    def __init__(self, crop_length):
        self.crop_length = crop_length

    def __call__(self, x):
        crop_bound = x.shape[0] - self.crop_length
        if crop_bound < 0:
            # edge pad if x is too short 
            pad_dur = (self.crop_length - len(x)) // 2 + 1 
            # print(f"X shape before pad: {x.shape}")
            x = np.pad(x, (pad_dur, pad_dur), "constant", constant_values=0 )
            # print(f"X shape after pad: {x.shape}")

            # re-compute crop bound
            crop_bound = x.shape[0] - self.crop_length
        start_idx = np.random.randint(crop_bound)
        x = x[start_idx:start_idx+self.crop_length]

        return x

class CenterCrop:
    def __init__(self, crop_length):
        self.crop_length = crop_length

    def __call__(self, x):
        sig_len = len(x)
        start_crop = int((sig_len - self.crop_length)/2)
        x = x[start_crop : start_crop + self.crop_length]
        return x

class CombineWithFixedDBSNR(torch.nn.Module):
    """
    Takes two signals and combines them at a specified dB SNR level.
    
    Returns:
        signal_in_noise, None 

    """
    def __init__(self):
        super(CombineWithFixedDBSNR, self).__init__()

    def forward(self, foreground_wav, background_wav, rand_db_snr):
        """
        Args:  
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as 
                the background audio sample
        """
        rms_ratio = np.power(10.0, rand_db_snr / 20.0)
        # Demean signal and noise before computing rms
        if foreground_wav is not None:
            foreground_wav = ch_demean(foreground_wav)
            rms_foreground = ch_rms(foreground_wav)
        else:
            rms_foreground = 0
            foreground_wav = torch.zeros(background_wav.shape)
        if background_wav is not None:
            background_wav = ch_demean(background_wav)
            rms_background = ch_rms(background_wav)
        else:
            rms_background = 0
            background_wav = torch.zeros(foreground_wav.shape)

        # Calculate the scale factor for the two sounds
        # For now, to align with the jsinv3 dataset, we include the infinite SNR 
        # cases
        if rms_foreground == 0: # No foreground condition (just noise)
            noise_scale_factor = 1
        elif rms_background == 0: 
            noise_scale_factor = 0
        else:
            noise_scale_factor = torch.div(rms_foreground, 
                                           torch.mul(rms_background,
                                                     rms_ratio))
 
        background_wav = torch.mul(noise_scale_factor, background_wav)
        signal_in_noise = torch.add(foreground_wav, background_wav)

        return signal_in_noise

class CombineWithRandomDBSNRWithParam(torch.nn.Module):
    """
    Takes two signals and combines them at a random dB SNR level, and returns 
    the selected dB SNR level (along side the combined signal).
    
    Args: 
        low_snr (float): the low end for the range of dB SNR to draw from
        high_snr (float): the high end for the range of db SNR to draw from
        rms_level (float): the end RMS value for the combined sound

    Returns:
        signal_in_noise, rand_db_snr 

    """
    def __init__(self, low_snr=-10, high_snr=10):
        self.low_snr=low_snr
        self.high_snr=high_snr
        super(CombineWithRandomDBSNRWithParam, self).__init__()

    def forward(self, foreground_wav, background_wav):
        """
        Args:  
            foreground_wav (torch.Tensor): the waveform that will be used as
                the foreground audio sample (usually speech)
            background_wav (torch.Tensor): the waveform that will be used as 
                the background audio sample
        """
        rand_db_snr = self.low_snr + (self.high_snr - self.low_snr) * torch.rand(1)
        rms_ratio = np.power(10.0, rand_db_snr / 20.0)
        # Demean signal and noise before computing rms
        if foreground_wav is not None:
            foreground_wav = ch_demean(foreground_wav)
            rms_foreground = ch_rms(foreground_wav)
        else:
            rms_foreground = 0
            foreground_wav = torch.zeros(background_wav.shape)
        if background_wav is not None:
            background_wav = ch_demean(background_wav)
            rms_background = ch_rms(background_wav)
        else:
            rms_background = 0
            background_wav = torch.zeros(foreground_wav.shape)

        # Calculate the scale factor for the two sounds
        # For now, to align with the jsinv3 dataset, we include the infinite SNR 
        # cases
        if rms_foreground == 0: # No foreground condition (just noise)
            noise_scale_factor = 1
        elif rms_background == 0: 
            noise_scale_factor = 0
        else:
            noise_scale_factor = torch.div(rms_foreground, 
                                           torch.mul(rms_background,
                                                     rms_ratio))
 
        background_wav = torch.mul(noise_scale_factor, background_wav)
        signal_in_noise = torch.add(foreground_wav, background_wav)

        return signal_in_noise, rand_db_snr


class MatchedCombineWithRandomDBSNR(torch.nn.Module):
    """
    Combines two signals at the same random dB SNR level.
    """
    def __init__(self, low_db=-10, high_db=10, return_param: bool=False, skip_aug_match: bool=False):
        super().__init__()
        self.low_db = low_db
        self.high_db = high_db
        self.return_param = return_param
        self.combiner_random = CombineWithRandomDBSNRWithParam(low_db, high_db)
        self.combiner_fixed = CombineWithFixedDBSNR()
        self.skip_aug_match = skip_aug_match

    def __call__(self, foreground_wav1, foreground_wav2, background_wav1, background_wav2):
        combined_1, rand_db_snr = self.combiner_random(foreground_wav1, background_wav1)
        if self.skip_aug_match:
            combined_2, rand_db_snr_2 = self.combiner_random(foreground_wav2, background_wav2)
        else:
            combined_2 =  self.combiner_fixed(foreground_wav2, background_wav2, rand_db_snr)
        if self.return_param:
            if self.skip_aug_match:
                return combined_1, combined_2, (float(rand_db_snr), float(rand_db_snr_2)) 
            else:
                return combined_1, combined_2, float(rand_db_snr)
        return combined_1, combined_2

class MatchedRandomSignalCrops(torch.nn.Module):
    """
    Randomly crops two signals to the same length, such that the word is in the same
    position (as defined by the center of the word) in both signals.
    """
    def __init__(self, crop_length=40000, skip_aug_match=False):
        super().__init__()
        self.crop_length = crop_length
        self.skip_aug_match = skip_aug_match

    def __call__(self, signal_1, signal_2):
        # assumes signal_1 is shorter than signal_2 (so crop_idx_1 is valid for signal_2)
        if isinstance(signal_2, (np.ndarray, torch.Tensor)):
            if len(signal_1) > len(signal_2):
                print('Warning: signal_1 is longer than signal_2 for matched Cropping')

        start_idx_1 = np.random.randint(signal_1.shape[0] - self.crop_length)
        cropped_1 = signal_1[start_idx_1:start_idx_1+self.crop_length]

        if signal_2 is None:
            return cropped_1, None

        # crop second signal so that the word is in the same start position
        if self.skip_aug_match:
            start_idx_2 = np.random.randint(signal_2.shape[0] - self.crop_length)
        else:
            start_idx_2 = start_idx_1 + (len(signal_2) - len(signal_1)) // 2
        cropped_2 = signal_2[start_idx_2:start_idx_2+self.crop_length]

        return cropped_1, cropped_2


class MatchedRandomSignalAugmentSox(torch.nn.Module):
    """
    Randomly applies the same set of signal augmentations to two signals. 
    Samples whether to apply filtering, pitch shifting, or tempo change 
    augmentations to signals, and the parameters for each augmentation. 
    """
    def __init__(self, sample_rate=20000, skip_aug_match=False):
        super().__init__()
        self.sample_rate = sample_rate
        self.skip_aug_match = skip_aug_match
    

    def __call__(self, signal_1, signal_2, speech=True, print_augments=False):
        ### Get shared augmentations:
        aug_dict = sample_augments(speech=speech)
        if print_augments:
            print(aug_dict)
        signal_1 = augment_excerpt(signal_1, aug_dict, sr=self.sample_rate)
        
        if self.skip_aug_match:
            # resample augmentations 
            aug_dict = sample_augments(speech=speech)

        signal_2 = augment_excerpt(signal_2, aug_dict, sr=self.sample_rate)

        return signal_1, signal_2
    
class ApplySingleAugmentSox(torch.nn.Module):
    """
    Applies the same class of signal augmentation to signals. 
    Either filtering, pitch shifting, or tempo change, determined 
    by augment_type. Samplesthe parameters for each augmentation. 
    """
    def __init__(self, augment_type, sample_rate=20000, return_params: bool=False):
        super().__init__()
        self.augment_type = augment_type
        self.sample_rate = sample_rate
        self.return_params = return_params

    def __call__(self, signal,  speech=True, print_augments=False):
        ### Get shared augmentations:
        aug_dict = sample_augments(speech=speech,
                                   effect_types=[self.augment_type],
                                   always_return_effect=True)
        if self.augment_type == 'filter':
            params = aug_dict['kwargs_butterworth']
            params = [params['order']] + params['cutoff']
        elif self.augment_type == 'pitch':
            params = aug_dict['kwargs_sox']['kwargs_pitch']['n_semitones']
        elif self.augment_type == 'tempo':
            params = aug_dict['kwargs_sox']['kwargs_tempo']['factor']

        if print_augments:
            print(aug_dict)
        signal = augment_excerpt(signal, aug_dict, sr=self.sample_rate)
        if self.return_params:
            return signal, params
        return signal

###################################
# Sox transforms for augmentations 
###################################


def loguniform(low, high, size=None):
    """
    Helper function to draw samples uniformly on a log scale.
    """
    return np.exp(np.random.uniform(low=np.log(low), high=np.log(high), size=size))


def pad_or_trim_to_len(x, n, mode='both', kwargs_pad={}):
    """
    Increases or decreases the length of a one-dimensional signal
    by either padding or triming the array. If the difference
    between `len(x)` and `n` is odd, this function will default to
    adding/removing the extra sample at the end of the signal.
    
    Args
    ----
    x (np.ndarray): one-dimensional input signal
    n (int): length of output signal
    mode (str): specify which end of signal to modify
        (default behavior is to symmetrically modify both ends)
    kwargs_pad (dict): keyword arguments for np.pad function
    
    Returns
    -------
    x_out (np.ndarray): one-dimensional signal with length `n`
    """
    assert len(np.array(x).shape) == 1, "input must be 1D array"
    assert mode.lower() in ['both', 'start', 'end']
    n_diff = np.abs(len(x) - n)
    if len(x) > n:
        if mode.lower() == 'end':
            x_out = x[:n]
        elif mode.lower() == 'start':
            x_out = x[-n:]
        else:
            x_out = x[int(np.floor(n_diff / 2)):-int(np.ceil(n_diff / 2))]
    elif len(x) < n:
        if mode.lower() == 'end':
            pad_width = [0, n_diff]
        elif mode.lower() == 'start':
            pad_width = [n_diff, 0]
        else:
            pad_width = [int(np.floor(n_diff / 2)), int(np.ceil(n_diff / 2))]
        kwargs = {'mode': 'constant'}
        kwargs.update(kwargs_pad)
        x_out = np.pad(x, pad_width, **kwargs)
    else:
        x_out = x
    assert len(x_out) == n
    return x_out

def apply_butterworth_filter(y,
                             sr,
                             order,
                             cutoff,
                             btype='bandpass',
                             mode='lfilter'):
    """
    """
    dtype = y.dtype
    b, a = scipy.signal.butter(
        order,
        cutoff,
        btype=btype,
        analog=False,
        output='ba',
        fs=sr)
    if mode.lower() == 'filtfilt':
        y_out = scipy.signal.filtfilt(b, a, y)
    elif mode.lower() == 'lfilter':
        y_out = scipy.signal.lfilter(b, a, y)
    else:
        raise ValueError("filter mode `{}` not recognized".format(mode))
    return y_out.astype(dtype)


def apply_sox_transformations(y, sr, kwargs_pitch={}, kwargs_tempo={}):
    """
    """
    dtype = y.dtype
    tfm = sox.Transformer()
    if kwargs_pitch:
        tfm.pitch(**kwargs_pitch)
    if kwargs_tempo:
        tfm.tempo(**kwargs_tempo)
    y_out = tfm.build_array(input_filepath=None, input_array=y, sample_rate_in=sr)
    y_out = pad_or_trim_to_len(y_out, len(y))
    return y_out.astype(dtype)


def augment_excerpt(y, aug_dict, sr=44100):
    """
    """
    kwargs_sox = aug_dict.get('kwargs_sox', {})
    if kwargs_sox:
        y_out = apply_sox_transformations(y, sr, **kwargs_sox)
        if np.isfinite(np.sqrt(np.mean(np.square(y_out)))):
            y = y_out
        else:
            print("[augment_excerpt] `apply_sox_transformations` produced Inf/NaN (skipped)")
    kwargs_butterworth = aug_dict.get('kwargs_butterworth', {})
    if kwargs_butterworth:
        y_out = apply_butterworth_filter(y, sr, **kwargs_butterworth)
        if np.isfinite(np.sqrt(np.mean(np.square(y_out)))):
            y = y_out
        else:
            print("[augment_excerpt] `apply_butterworth_filter` produced Inf/NaN (skipped)")
    return y


def sample_augments(speech=True,
                    sample_rate=20_000,
                    effect_types = ['filter', 'pitch',  'tempo'],
                    always_return_effect = False):
    """
    """
    # Sample bandpass filter parameters
    if always_return_effect:
        effect_choice = effect_types
    else:
        n_effects = np.random.randint(low=0, high=len(effect_types)+1)
        effect_choice = np.random.choice(effect_types, size=n_effects, replace=False)
    # sample if using filtering
    aug_dict = {}
    if 'filter' in effect_choice:
        nyquist = (sample_rate // 2) - 1 # limit must be exactly under and not equal to for filtering 
        if speech:
            ## Use frequency ranges that overlap speech signals 
            range_bandpass_freq_low = [4e1, 4e2]
            range_bandpass_freq_high = [4e3, 10e3]
            bandpass_freq_low = loguniform(*range_bandpass_freq_low)
            bandpass_freq_high = loguniform(*range_bandpass_freq_high)
        else:
            ## Can use wider frequency range for non-speech signals
            range_bandpass_center_frequency = [16e1, 10e3]
            range_bandpass_bandwidth_octave = [2, 4]
            bandpass_center_frequency = loguniform(*range_bandpass_center_frequency)
            bandpass_bandwidth_octave = loguniform(*range_bandpass_bandwidth_octave)
            bandpass_freq_low = np.power(2, -bandpass_bandwidth_octave/2) * bandpass_center_frequency
            bandpass_freq_high = np.power(2, bandpass_bandwidth_octave/2) * bandpass_center_frequency
        # sample order for nth order butterworth filter 
        list_bandpass_order = [1, 2, 3, 4]
        bandpass_order_int = np.random.choice(list_bandpass_order)
        # clip hz based on nyquist 
        if bandpass_freq_low < 20:
            bandpass_freq_low = 20 
        if bandpass_freq_high > nyquist:
            bandpass_freq_high = nyquist 

        # stack as kwargs dict 
        aug_dict["kwargs_butterworth"] = {'order': bandpass_order_int,
                            'cutoff': [bandpass_freq_low, bandpass_freq_high],
                            'btype': 'bandpass'
                            }

    # Sample sox augmentations
    dict_kwargs_sox = {}
    for effect in effect_choice:
        if effect == 'pitch':
            pitch_n_semitones = np.random.uniform(-0.5, 0.5)
            # pitch args are n semitones
            dict_kwargs_sox['kwargs_pitch'] = {
                'n_semitones': pitch_n_semitones,
                'quick': False,
            }
        elif effect == 'tempo':
            tempo_factor = np.random.uniform(0.80, 1.20)
            # tempo args are factor
            dict_kwargs_sox['kwargs_tempo'] = {
                            'factor': tempo_factor,
                            'audio_type': 's',
                            'quick': False,
            }
            
    aug_dict['kwargs_sox'] = dict_kwargs_sox

    return aug_dict

