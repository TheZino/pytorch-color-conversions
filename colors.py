from warnings import warn

import numpy as np
import torch
from scipy import linalg

xyz_from_rgb = torch.Tensor([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])

rgb_from_xyz = torch.Tensor(linalg.inv(xyz_from_rgb))

illuminants = \
    {"A": {'2': torch.Tensor([(1.098466069456375, 1, 0.3558228003436005)]),
           '10': torch.Tensor([(1.111420406956693, 1, 0.3519978321919493)])},
     "D50": {'2': torch.Tensor([(0.9642119944211994, 1, 0.8251882845188288)]),
             '10': torch.Tensor([(0.9672062750333777, 1, 0.8142801513128616)])},
     "D55": {'2': torch.Tensor([(0.956797052643698, 1, 0.9214805860173273)]),
             '10': torch.Tensor([(0.9579665682254781, 1, 0.9092525159847462)])},
     "D65": {'2': torch.Tensor([(0.95047, 1., 1.08883)]),   # This was: `lab_ref_white`
             '10': torch.Tensor([(0.94809667673716, 1, 1.0730513595166162)])},
     "D75": {'2': torch.Tensor([(0.9497220898840717, 1, 1.226393520724154)]),
             '10': torch.Tensor([(0.9441713925645873, 1, 1.2064272211720228)])},
     "E": {'2': torch.Tensor([(1.0, 1.0, 1.0)]),
           '10': torch.Tensor([(1.0, 1.0, 1.0)])}}


# -------------------------------------------------------------
# The conversion functions that make use of the matrices above
# -------------------------------------------------------------

def _convert(matrix, arr):
    """Do the color space conversion.
    Parameters
    ----------
    matrix : array_like
        The 3x3 matrix to use.
    arr : array_like
        The input array.
    Returns
    -------
    out : ndarray, dtype=float
        The converted array.
    """

    if arr.is_cuda:
        matrix = matrix.cuda()

    bs, ch, h, w = arr.shape

    arr = arr.permute((0,2,3,1))
    arr = arr.contiguous().view(-1,1,3)

    matrix = matrix.transpose(0,1).unsqueeze(0)
    matrix = matrix.repeat(arr.shape[0],1,1)

    res = torch.bmm(arr,matrix)

    res = res.view(bs,h,w,ch)
    res = res.transpose(3,2).transpose(2,1)


    return res

def get_xyz_coords(illuminant, observer):
    """Get the XYZ coordinates of the given illuminant and observer [1]_.
    Parameters
    ----------
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    (x, y, z) : tuple
        A tuple with 3 elements containing the XYZ coordinates of the given
        illuminant.
    Raises
    ------
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    """
    illuminant = illuminant.upper()
    try:
        return illuminants[illuminant][observer]
    except KeyError:
        raise ValueError("Unknown illuminant/observer combination\
        (\'{0}\', \'{1}\')".format(illuminant, observer))


##### RGB - LAB


def rgb2xyz(rgb):

    mask = rgb > 0.04045
    rgbm = rgb.clone()
    tmp = torch.pow((rgb + 0.055) / 1.055, 2.4)
    rgb = torch.where(mask, tmp, rgb)

    rgbm = rgb.clone()
    rgb[~mask] = rgbm[~mask]/12.92
    return _convert(xyz_from_rgb, rgb)

def xyz2lab(xyz, illuminant="D65", observer="2"):

    # arr = _prepare_colorarray(xyz)
    xyz_ref_white = get_xyz_coords(illuminant, observer)
    #cuda
    if xyz.is_cuda:
        xyz_ref_white = xyz_ref_white.cuda()

    # scale by CIE XYZ tristimulus values of the reference white point
    xyz = xyz / xyz_ref_white.view(1,3,1,1)
    # Nonlinear distortion and linear transformation
    mask = xyz > 0.008856
    xyzm = xyz.clone()
    xyz[mask] = torch.pow(xyzm[mask], 1/3)
    xyzm = xyz.clone()
    xyz[~mask] = 7.787 * xyzm[~mask] + 16. / 116.
    x, y, z = xyz[:, 0, :, :], xyz[:, 1, :, :], xyz[:, 2, :, :]
    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    return torch.stack((L,a,b), 1)

def rgb2lab(rgb, illuminant="D65", observer="2"):
    """RGB to lab color space conversion.
    Parameters
    ----------
    rgb : torch.Tensor
        The image in RGB format, in a 3- or 4-D array of shape
        ``(batch_size x 3 x h x w)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    out : torch.Tensor
        The image in Lab format, in a 3- or 4-D array of shape
        ``(batch_size x 3 x h x w)``.
    Notes
    -----
    This function uses rgb2xyz and xyz2lab.
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.
    """
    return xyz2lab(rgb2xyz(rgb), illuminant, observer)



##### LAB - RGB


def lab2xyz(lab, illuminant="D65", observer="2"):
    arr = lab.clone()
    L, a, b = arr[:, 0, :, :], arr[:, 1, :, :], arr[:, 2, :, :]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    if (z < 0).sum() > 0:
        invalid = np.nonzero(z < 0)
        warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
        z[invalid] = 0

    out = torch.stack((x, y, z),1)

    mask = out > 0.2068966
    outm = out.clone()
    out[mask] = torch.pow(outm[mask], 3.)
    outm = out.clone()
    out[~mask] = (outm[~mask] - 16.0 / 116.) / 7.787

    # rescale to the reference white (illuminant)
    xyz_ref_white = get_xyz_coords(illuminant, observer)
    # cuda
    if lab.is_cuda:
        xyz_ref_white = xyz_ref_white.cuda()
    xyz_ref_white = xyz_ref_white.unsqueeze(2).unsqueeze(2).repeat(1,1,out.shape[2],out.shape[3])
    out = out * xyz_ref_white
    return out

def xyz2rgb(xyz):
    arr = _convert(rgb_from_xyz, xyz)
    mask = arr > 0.0031308
    arrm = arr.clone()
    arr[mask] = 1.055 * torch.pow(arrm[mask], 1 / 2.4) - 0.055
    arrm = arr.clone()
    arr[~mask] = arrm[~mask] * 12.92

    mask_z = arr < 0
    arr[mask_z] = 0
    mask_o = arr > 1
    arr[mask_o] = 1

    return arr

def lab2rgb(lab, illuminant="D65", observer="2"):
    """Lab to RGB color space conversion.
    Parameters
    ----------
    lab : torch.Tensor
        The image in Lab format, in a 3-D array of shape
        ``(batch_size x 3 x h x w)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    out : torch.Tensor
        The image in RGB format, in a 3-D array of shape
        ``(batch_size x 3 x h x w)``.
    Notes
    -----
    This function uses lab2xyz and xyz2rgb.
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.
    """
    return xyz2rgb(lab2xyz(lab, illuminant, observer))
