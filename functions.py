import numpy as np 
from skimage.transform import resize,rotate

def circle_mask(y,x):
    '''
    Mask using in iradon_transform.

    Parameters
    ----------
    y : int
        y size.
    x : int
        x size.

    Returns
    -------
    mask : array
        mask to remove artefacts.

    '''
    Y, X = np.ogrid[:y, :x]
    mask = np.sqrt((X-x//2)**2+(Y-y//2)**2)
    rad = x//2
    mask = mask<=rad
    return mask

def radon_transform(img,n_angles):
    '''
    Creating sinogram from 2d image.

    Parameters
    ----------
    img : array
        input 2d image.
    n_angles : int
        values of angles.

    Returns
    -------
    sinogram : array
        returns 2d sinogram.

    '''
    min_angle = 180/2/n_angles
    max_angle = 180-180/2/n_angles
    theta = np.linspace(min_angle,max_angle,n_angles)
    sinogram = np.zeros((len(img),n_angles))
    for i in range(np.amin(np.shape(sinogram))):
        rot_image = rotate(img,theta[i])
        sinogram[:,i] = np.flip(np.squeeze(resize(rot_image,(1,len(img)))))
    return sinogram

def iradon_transform(sinogram,n_angles):
    '''
    Function to reconstruct image from sinogram.

    Parameters
    ----------
    sinogram : array
        sinogram to reconstruct.
    n_angles : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    ### CREATE RAMP FILTER
    win = np.zeros(len(sinogram)//2)
    win[:len(win)//2] = np.arange(1,len(win)+1,2)
    win[len(win)//2:] = np.arange(len(win)-1,0,-2)
    fft_filter = np.zeros(len(sinogram))
    fft_filter[0] = 0.25
    fft_filter[1::2] = -1/(np.pi*win)**2
    ramp_filter = np.real(np.fft.fft(fft_filter))*2
    ### FILTERING AND IRADON
    out = np.zeros((len(sinogram), len(sinogram)))
    min_angle = 180/2/n_angles
    max_angle = 180-180/2/n_angles
    theta = np.linspace(min_angle,max_angle,n_angles)
    for i in range(n_angles):
        fft =(np.fft.fft(sinogram[:,i]))*ramp_filter
        fft = np.real(np.fft.ifft(fft))
        sinogram_line = resize(fft,(len(out),len(out)))
        sinogram_line= rotate(sinogram_line,theta[i])*circle_mask(len(out), len(out))
        out = out+sinogram_line   
    out = np.rot90(out)
    out = out*np.pi/(2*n_angles)
    return out