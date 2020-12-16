# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 22:07:52 2020

@author: weicc
"""
from __future__ import division
from numpy import sqrt, mean, absolute, real, conj


def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return sqrt(mean(absolute(a)**2))


def rms_fft(spectrum):
    """
    Use Parseval's theorem to find the RMS value of a signal from its fft,
    without wasting time doing an inverse FFT.
    For a signal x, these should produce the same result, to within numerical
    accuracy:
    rms_flat(x) ~= rms_fft(fft(x))
    """
    return rms_flat(spectrum)/sqrt(len(spectrum))


def rms_rfft(spectrum, n=None):
    """
    Use Parseval's theorem to find the RMS value of an even-length signal
    from its rfft, without wasting time doing an inverse real FFT.
    spectrum is produced as spectrum = numpy.fft.rfft(signal)
    For a signal x with an even number of samples, these should produce the
    same result, to within numerical accuracy:
    rms_flat(x) ~= rms_rfft(rfft(x))
    If len(x) is odd, n must be included, or the result will only be
    approximate, due to the ambiguity of rfft for odd lengths.
    """
    if n is None:
        n = (len(spectrum) - 1) * 2
    sq = real(spectrum * conj(spectrum))
    if n % 2:  # odd-length
        mean = (sq[0] + 2*sum(sq[1:])           )/n
    else:  # even-length
        mean = (sq[0] + 2*sum(sq[1:-1]) + sq[-1])/n
    root = sqrt(mean)
    return root/sqrt(n)

def selfrms_fft(spectrum):
    temp = 0
    for i in range(len(spectrum)):
        temp = temp + abs(spectrum[i]**2)
        
    return sqrt(temp)/15000


def halfselfrms_fft(spectrum):
    temp = 0
    for i in range((int)(len(spectrum)/2)):
        temp = temp + abs(spectrum[i]**2)
        
    return sqrt(2*temp)/15000

if __name__ == "__main__":
    from numpy.random import randn
    from numpy.fft import fft, ifft, rfft, irfft
    import numpy 
    
    f = 100
    N = 15000
    
    t = numpy.linspace(0,1,N)
    x = 1.414 * numpy.sin(2*numpy.pi*f*t)
    X = fft(x)
    
#    n = 17
#    x = randn(n)
#    X = fft(x)
#    rX = rfft(x)

    rms = numpy.sqrt(numpy.mean(x**2))
    
    print(halfselfrms_fft(X))
    print(selfrms_fft(X))
    print (rms_flat(x))
    print (rms_flat(ifft(X)))
    print (rms_fft(X))


    # Accurate for odd n:
#    print( rms_flat(irfft(rX, n)))
#    print( rms_rfft(rX, n))


    # Only approximate for odd n:
#    print( rms_flat(irfft(rX)))
#    print( rms_rfft(rX))