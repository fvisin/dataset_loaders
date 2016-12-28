import numpy as np
from scipy import interpolate
import scipy.ndimage as ndi
import SimpleITK as sitk
from skimage.color import rgb2gray, gray2rgb
from skimage import img_as_float
import seaborn as sns
import os
import scipy.misc


# Converts a label mask to RGB to be shown
def my_label2rgb(labels, colors, bglabel=None, bg_color=(0., 0., 0.)):
    output = np.zeros(labels.shape + (3,), dtype=np.float64)
    for i in range(len(colors)):
        if i != bglabel:
            output[(labels == i).nonzero()] = colors[i]
    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color
    return output


# Converts a label mask to RGB to be shown and overlaps over an image
def my_label2rgboverlay(labels, colors, image, bglabel=None,
                        bg_color=(0., 0., 0.), alpha=0.2):
    image_float = gray2rgb(img_as_float(rgb2gray(image)))
    label_image = my_label2rgb(labels, colors, bglabel=bglabel,
                               bg_color=bg_color)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


# Save 2 images (Image and mask)
def save_img2(img, mask, fname, color_map, void_label):
    img = img.transpose((1, 2, 0))
    label_mask = my_label2rgboverlay(mask,
                                     colors=color_map,
                                     image=img,
                                     bglabel=void_label,
                                     alpha=0.2)
    combined_image = np.concatenate((img, label_mask),
                                    axis=1)
    scipy.misc.toimage(combined_image).save(fname)


def displacement_vecs(std, grid_size):
    disp_x = np.reshape(
        map(int, np.random.randn(np.prod(np.asarray(grid_size)))*std),
        grid_size)
    disp_y = np.reshape(
        map(int, np.random.randn(np.prod(np.asarray(grid_size)))*std),
        grid_size)
    return disp_x, disp_y


def _elastic_def(image, fx, fy):
    imsize = image.shape[-2:]
    x = np.arange(imsize[0]-1)
    y = np.arange(imsize[1]-1)
    xx, yy = np.meshgrid(x, y)
    z_1 = fx(x, y).astype(int)
    z_2 = fy(x, y).astype(int)
    img = np.zeros(image.shape, dtype='uint8')
    if image.ndim == 3:  # multi-channel
        img[:, xx, yy] = image[:, np.clip(xx - z_1, 0, imsize[0]-1),
                               np.clip(yy - z_2, 0, imsize[1]-1)]
    if image.ndim == 2:  # one channel
        img[xx, yy] = image[np.clip(xx - z_1, 0, imsize[0]-1),
                            np.clip(yy - z_2, 0, imsize[1]-1)]
    return img


def batch_elastic_def(im_batch, disp_x, disp_y, grid_size=(3, 3),
                      interpol_kind='linear'):
    '''im_batch should have batch dimension
       even if it contains a single image'''
    imsize = im_batch.shape[-2:]
    x = np.linspace(0, imsize[0]-1, grid_size[0]).astype(int)
    y = np.linspace(0, imsize[1]-1, grid_size[1]).astype(int)
    fx = interpolate.interp2d(x, y, disp_x, kind=interpol_kind)
    fy = interpolate.interp2d(x, y, disp_y, kind=interpol_kind)
    if im_batch.ndim == 3 or im_batch.ndim == 4:
        im_elast_def = np.asarray([_elastic_def(im, fx, fy)
                                   for im in im_batch])
    if im_batch.ndim == 2:  # if no batch in im_batch
        im_elast_def = _elastic_def(im_batch, fx, fy)
    return im_elast_def


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, fill_mode='nearest', cval=0.,
                    order=0):
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    # Apply the transformation on each channel (sequence, batch, ..)
    for i in range(x.shape[0]):
        x[i] = ndi.interpolation.affine_transform(x[i], final_affine_matrix,
                                                  final_offset, order=order,
                                                  mode=fill_mode, cval=cval)
    return x


def random_channel_shift(x_in, intensity, channel_index=0):
    x_out = np.zeros(x_in.shape, dtype=x_in.dtype)
    for i, x in enumerate(x_in):
        x = np.rollaxis(x, channel_index, 0)
        min_x, max_x = np.min(x), np.max(x)
        channel_images = [np.clip(x_channel + np.random.uniform(-intensity,
                                                                intensity),
                                  min_x, max_x)
                          for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x_out[i] = np.rollaxis(x, 0, channel_index+1)
    return x_out


def flip_axis(x_in, axis):
    x_out = np.zeros(x_in.shape, dtype=x_in.dtype)
    for i, x in enumerate(x_in):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x_out[i] = x.swapaxes(0, axis)
    return x_out


# Define warp
def gen_warp_field(shape, sigma=0.1, grid_size=3):
    # Initialize bspline transform
    args = shape+(sitk.sitkFloat32,)
    ref_image = sitk.Image(*args)
    tx = sitk.BSplineTransformInitializer(ref_image, [grid_size, grid_size])

    # Initialize shift in control points:
    # mesh size = number of control points - spline order
    p = sigma * np.random.randn(grid_size+3, grid_size+3, 2)

    # Anchor the edges of the image
    p[:, 0, :] = 0
    p[:, -1:, :] = 0
    p[0, :, :] = 0
    p[-1:, :, :] = 0

    # Set bspline transform parameters to the above shifts
    tx.SetParameters(p.flatten())

    # Compute deformation field
    displacement_filter = sitk.TransformToDisplacementFieldFilter()
    displacement_filter.SetReferenceImage(ref_image)
    displacement_field = displacement_filter.Execute(tx)

    return displacement_field


# Pad image
def pad_image(x, pad_amount, mode='reflect', constant=0.):
    e = pad_amount
    shape = list(x.shape)
    shape[:2] += 2*e
    if mode == 'constant':
        x_padded = np.ones(shape, dtype=np.float32)*constant
        x_padded[e:-e, e:-e] = x.copy()
    else:
        x_padded = np.zeros(shape, dtype=np.float32)
        x_padded[e:-e, e:-e] = x.copy()

    if mode == 'reflect':
        # Edges
        x_padded[:e, e:-e] = np.flipud(x[:e, :])  # left
        x_padded[-e:, e:-e] = np.flipud(x[-e:, :])  # right
        x_padded[e:-e, :e] = np.fliplr(x[:, :e])  # top
        x_padded[e:-e, -e:] = np.fliplr(x[:, -e:])  # bottom
        # Corners
        x_padded[:e, :e] = np.fliplr(np.flipud(x[:e, :e]))  # top-left
        x_padded[-e:, :e] = np.fliplr(np.flipud(x[-e:, :e]))  # top-right
        x_padded[:e, -e:] = np.fliplr(np.flipud(x[:e, -e:]))  # bottom-left
        x_padded[-e:, -e:] = np.fliplr(np.flipud(x[-e:, -e:]))  # bottom-right
    elif mode == 'zero' or mode == 'constant':
        pass
    elif mode == 'nearest':
        # Edges
        x_padded[:e, e:-e] = x[[0], :]  # left
        x_padded[-e:, e:-e] = x[[-1], :]  # right
        x_padded[e:-e, :e] = x[:, [0]]  # top
        x_padded[e:-e, -e:] = x[:, [-1]]  # bottom
        # Corners
        x_padded[:e, :e] = x[[0], [0]]  # top-left
        x_padded[-e:, :e] = x[[-1], [0]]  # top-right
        x_padded[:e, -e:] = x[[0], [-1]]  # bottom-left
        x_padded[-e:, -e:] = x[[-1], [-1]]  # bottom-right
    else:
        raise ValueError("Unsupported padding mode \"{}\"".format(mode))
    return x_padded


def apply_warp(x, warp_field, fill_mode='reflect',
               interpolator=sitk.sitkLinear,
               fill_constant=0):
    # Expand deformation field (and later the image), padding for the largest
    # deformation
    warp_field_arr = sitk.GetArrayFromImage(warp_field)
    max_deformation = np.max(np.abs(warp_field_arr))
    pad = np.ceil(max_deformation).astype(np.int32)
    warp_field_padded_arr = pad_image(warp_field_arr, pad_amount=pad,
                                      mode='nearest')
    warp_field_padded = sitk.GetImageFromArray(warp_field_padded_arr,
                                               isVector=True)

    # Warp x, one filter slice at a time
    x_warped = np.zeros(x.shape, dtype=np.float32)
    warp_filter = sitk.WarpImageFilter()
    warp_filter.SetInterpolator(interpolator)
    warp_filter.SetEdgePaddingValue(np.min(x).astype(np.double))
    for i, image in enumerate(x):
        x_tmp = np.zeros(image.shape, dtype=image.dtype)
        for j, channel in enumerate(image):
            image_padded = pad_image(channel, pad_amount=pad, mode=fill_mode,
                                     constant=fill_constant).T
            image_f = sitk.GetImageFromArray(image_padded)
            image_f_warped = warp_filter.Execute(image_f, warp_field_padded)
            image_warped = sitk.GetArrayFromImage(image_f_warped)
            x_tmp[j] = image_warped[pad:-pad, pad:-pad].T
        x_warped[i] = x_tmp
    return x_warped


def random_transform(x, y=None,
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     cvalMask=0.,
                     horizontal_flip=0.,  # probability
                     vertical_flip=0.,  # probability
                     rescale=None,
                     spline_warp=False,
                     warp_sigma=0.1,
                     warp_grid_size=3,
                     crop_size=None,
                     nclasses=None,
                     gamma=0.,
                     gain=1.,
                     chan_idx=3,  # No batch yet: (s, 0, 1, c)
                     rows_idx=1,  # No batch yet: (s, 0, 1, c)
                     cols_idx=2,  # No batch yet: (s, 0, 1, c)
                     void_label=None):

    # Set this to a dir, if you want to save augmented images samples
    save_to_dir = None

    # listify zoom range
    if np.isscalar(zoom_range):
        zoom_range = [1 - zoom_range, 1 + zoom_range]
    elif len(zoom_range) == 2:
        zoom_range = list(zoom_range)
    else:
        raise Exception('zoom_range should be a float or '
                        'a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    # Channel shift
    if channel_shift_range != 0:
        x = random_channel_shift(x, channel_shift_range, chan_idx)

    # Reshape to (anything_else, 0, 1)
    pattern = [el for el in range(x.ndim) if el != rows_idx and el != cols_idx]
    pattern += [rows_idx, cols_idx]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)
    x_shape = list(x.shape)
    x = x.reshape([-1] + x_shape[-2:])  # squash everything on channels
    if y is not None and len(y) > 0:
        y = y.transpose(pattern)
        y_shape = list(y.shape)
        y = y.reshape([-1] + y_shape[-2:])  # squash everything on channels
    chan_idx = None
    rows_idx = 1
    cols_idx = 2

    # Gamma correction
    if (gamma > 0 or rotation_range or height_shift_range or
            width_shift_range or shear_range or zoom_range != [1, 1]):
        if gamma > 0:
            scale = float(1)
            x = ((x / scale) ** gamma) * scale * gain

        # Affine transformations (zoom, rotation, shift, ..)
        # --> Rotation
        if rotation_range:
            theta = np.pi / 180 * np.random.uniform(-rotation_range,
                                                    rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        # --> Shift/Translation
        if height_shift_range:
            tx = (np.random.uniform(-height_shift_range, height_shift_range) *
                  x.shape[rows_idx])
        else:
            tx = 0
        if width_shift_range:
            ty = (np.random.uniform(-width_shift_range, width_shift_range) *
                  x.shape[cols_idx])
        else:
            ty = 0
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        # --> Shear
        if shear_range:
            shear = np.random.uniform(-shear_range, shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        # --> Zoom
        if zoom_range == [1, 1]:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        # Use a composition of homographies to generate the final transform
        # that has to be applied
        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix,
                                                translation_matrix),
                                         shear_matrix), zoom_matrix)
        h, w = x.shape[rows_idx], x.shape[cols_idx]
        transform_matrix = transform_matrix_offset_center(transform_matrix,
                                                          h, w)
        # Apply all the transformations together
        x = apply_transform(x, transform_matrix, fill_mode=fill_mode,
                            cval=cval, order=1)
        if y is not None and len(y) > 0:
            y = apply_transform(y, transform_matrix, fill_mode=fill_mode,
                                cval=cvalMask, order=0)

    # Horizontal flip
    if np.random.random() < horizontal_flip:  # 0 = disabled
        x = flip_axis(x, cols_idx)
        if y is not None and len(y) > 0:
            y = flip_axis(y, cols_idx)

    # Vertical flip
    if np.random.random() < vertical_flip:  # 0 = disabled
        x = flip_axis(x, rows_idx)
        if y is not None and len(y) > 0:
            y = flip_axis(y, rows_idx)

    # Spline warp
    if spline_warp:
        warp_field = gen_warp_field(shape=x.shape[-2:],
                                    sigma=warp_sigma,
                                    grid_size=warp_grid_size)
        x = apply_warp(x, warp_field,
                       interpolator=sitk.sitkLinear,
                       fill_mode=fill_mode,
                       fill_constant=cval)
        if y is not None and len(y) > 0:
            y = np.round(apply_warp(y, warp_field,
                                    interpolator=sitk.sitkNearestNeighbor,
                                    fill_mode=fill_mode,
                                    fill_constant=cvalMask))

    # Crop
    # Expects axes with shape (..., 0, 1)
    if crop_size:
        crop = list(crop_size)
        pad = [0, 0]
        h, w = x.shape[rows_idx], x.shape[cols_idx]

        # Compute amounts
        if crop[0] < h:
            # Do random crop
            top = np.random.randint(h - crop[0])
        else:
            # Set pad and reset crop
            pad[0] = crop[0] - h
            top, crop[0] = 0, h
        if crop[1] < w:
            # Do random crop
            left = np.random.randint(w - crop[1])
        else:
            # Set pad and reset crop
            pad[1] = crop[1] - w
            left, crop[1] = 0, w

        # Cropping
        x = x[..., top:top+crop[0], left:left+crop[1]]
        if y is not None and len(y) > 0:
            y = y[..., top:top+crop[0], left:left+crop[1]]
        # Padding
        if pad != [0, 0]:
            pad_pattern = ((0, 0),) * (x.ndim - 2) + (
                (pad[0]//2, pad[0] - pad[0]//2),
                (pad[1]//2, pad[1] - pad[1]//2))
            x = np.pad(x, pad_pattern, 'constant')
            y = np.pad(y, pad_pattern, 'constant', constant_values=void_label)

    # Save augmented images
    if save_to_dir:
        fname = 'data_augm_{}.png'.format(np.random.randint(1e4))
        print ('Save to dir'.format(fname))
        color_map = sns.hls_palette(nclasses)
        save_img2(x[0], y[0, 0, :, :], os.path.join(save_to_dir, fname),
                  color_map, void_label)

    x_shape[-2] = x.shape[rows_idx]
    x_shape[-1] = x.shape[cols_idx]
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    if y is not None and len(y) > 0:
        y_shape[-2] = y.shape[rows_idx]
        y_shape[-1] = y.shape[cols_idx]
        y = y.reshape(y_shape)  # unsquash
        y = y.transpose(inv_pattern)
    return x, y
