# Based on
# https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
import os

import numpy as np
import scipy.misc
import scipy.ndimage as ndi
from skimage.color import rgb2gray, gray2rgb
from skimage import img_as_float


def optical_flow(seq, rows_idx, cols_idx, chan_idx, return_rgb=False):
    '''Optical flow

    Takes a 4D array of sequences and returns a 4D array with
    an RGB optical flow image for each frame in the input'''
    import cv2
    if seq.ndim != 4:
        raise RuntimeError('Optical flow expected 4 dimensions, got %d' %
                           seq.ndim)
    seq = seq.copy()
    seq = (seq * 255).astype('uint8')
    # Reshape to channel last: (b*seq, 0, 1, ch) if seq
    pattern = [el for el in range(seq.ndim)
               if el not in (rows_idx, cols_idx, chan_idx)]
    pattern += [rows_idx, cols_idx, chan_idx]
    inv_pattern = [pattern.index(el) for el in range(seq.ndim)]
    seq = seq.transpose(pattern)
    if seq.shape[0] == 1:
        raise RuntimeError('Optical flow needs a sequence longer than 1 '
                           'to work')
    seq = seq[..., ::-1]  # Go BGR for OpenCV

    frame1 = seq[0]
    if return_rgb:
        flow_seq = np.zeros_like(seq)
        hsv = np.zeros_like(frame1)
    else:
        sh = list(seq.shape)
        sh[-1] = 2
        flow_seq = np.zeros(sh)

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Go to gray

    flow = None
    for i, frame2 in enumerate(seq[1:]):
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # Go to gray
        flow = cv2.calcOpticalFlowFarneback(prev=frame1,
                                            next=frame2,
                                            pyr_scale=0.5,
                                            levels=3,
                                            winsize=10,
                                            iterations=3,
                                            poly_n=5,
                                            poly_sigma=1.1,
                                            flags=0,
                                            flow=flow)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1],
                                   angleInDegrees=True)
        # normalize between 0 and 255
        ang = ang / 360 * 255
        if return_rgb:
            hsv[..., 0] = ang
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            flow_seq[i+1] = rgb
            # Image.fromarray(rgb).show()
            # cv2.imwrite('opticalfb.png', frame2)
            # cv2.imwrite('opticalhsv.png', bgr)
        else:
            flow_seq[i+1] = np.stack((ang, mag), 2)
        frame1 = frame2
    flow_seq = flow_seq.transpose(inv_pattern)
    return flow_seq / 255.  # return in [0, 1]


def my_label2rgb(labels, cmap, bglabel=None, bg_color=(0., 0., 0.)):
    '''Convert a label mask to RGB applying a color map'''
    output = np.zeros(labels.shape + (3,), dtype=np.float64)
    for i in range(len(cmap)):
        if i != bglabel:
            output[(labels == i).nonzero()] = cmap[i]
    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color
    return output


def my_label2rgboverlay(labels, cmap, image, bglabel=None,
                        bg_color=(0., 0., 0.), alpha=0.2):
    '''Superimpose a mask over an image

    Convert a label mask to RGB applying a color map and superimposing it
    over an image as a transparent overlay'''
    image_float = gray2rgb(img_as_float(rgb2gray(image)))
    label_image = my_label2rgb(labels, cmap, bglabel=bglabel,
                               bg_color=bg_color)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


def save_img2(x, y, fname, cmap, void_label, rows_idx, cols_idx,
              chan_idx):
    '''Save a mask and an image side to side

    Convert a label mask to RGB applying a color map and superimposing it
    over an image as a transparent overlay. Saves the original image and
    the image with the mask overlay in a file'''
    pattern = [el for el in range(x.ndim) if el not in [rows_idx, cols_idx,
                                                        chan_idx]]
    pattern += [rows_idx, cols_idx, chan_idx]

    x_copy = x.transpose(pattern)
    if y is not None and len(y) > 0:
        y_copy = y.transpose(pattern)

    # Take only the first batch
    x_copy = x_copy[0]
    if y is not None and len(y) > 0:
        # Take only the first batch and drop extra dim
        y_copy = y_copy[0, ..., 0]
        label_mask = my_label2rgboverlay(y_copy,
                                         cmap=cmap,
                                         image=x_copy,
                                         bglabel=void_label,
                                         alpha=0.2)
        combined_image = np.concatenate((x_copy, label_mask), axis=1)
    else:
        combined_image = x_copy
    scipy.misc.toimage(combined_image).save(fname)


def transform_matrix_offset_center(matrix, x, y):
    '''Shift the transformation matrix to be in the center of the image

    Apply an offset to the transformation matrix so that the origin of
    the axis is in the center of the image.'''
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, fill_mode='nearest', cval=0.,
                    order=0, rows_idx=1, cols_idx=2):
    '''Apply an affine transformation on each channel separately.'''
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    # Reshape to (*, 0, 1)
    pattern = [el for el in range(x.ndim) if el != rows_idx and el != cols_idx]
    pattern += [rows_idx, cols_idx]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)
    x_shape = list(x.shape)
    x = x.reshape([-1] + x_shape[-2:])  # squash everything on the first axis

    # Apply the transformation on each channel, sequence, batch, ..
    for i in range(x.shape[0]):
        x[i] = ndi.interpolation.affine_transform(x[i], final_affine_matrix,
                                                  final_offset, order=order,
                                                  mode=fill_mode, cval=cval)
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    return x


def random_channel_shift(x, shift_range, rows_idx, cols_idx, chan_idx):
    '''Shift the intensity values of each channel uniformly.

    Channel by channel, shift all the intensity values by a random value in
    [-shift_range, shift_range]'''
    pattern = [chan_idx]
    pattern += [el for el in range(x.ndim) if el not in [rows_idx, cols_idx,
                                                         chan_idx]]
    pattern += [rows_idx, cols_idx]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)  # channel first
    x_shape = list(x.shape)
    # squash rows and cols together and everything else on the 1st
    x = x.reshape((-1, x_shape[-2] * x_shape[-1]))
    # Loop on the channels/batches/etc
    for i in range(x.shape[0]):
        min_x, max_x = np.min(x), np.max(x)
        x[i] = np.clip(x[i] + np.random.uniform(-shift_range, shift_range),
                       min_x, max_x)
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    return x


def flip_axis(x, flipping_axis):
    '''Flip an axis by inverting the position of its elements'''
    pattern = [flipping_axis]
    pattern += [el for el in range(x.ndim) if el != flipping_axis]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)  # "flipping_axis" first
    x = x[::-1, ...]
    x = x.transpose(inv_pattern)
    return x


def pad_image(x, pad_amount, mode='reflect', constant=0.):
    '''Pad an image

    Pad an image by pad_amount on each side.

    Parameters
    ----------
    x: numpy ndarray
        The array to be padded.
    pad_amount: int
        The number of pixels of the padding.
    mode: string
        The padding mode. If "constant" a constant value will be used to
        fill the padding; if "reflect" the border pixels will be used in
        inverse order to fill the padding; if "nearest" the border pixel
        closer to the padded area will be used to fill the padding; if
        "zero" the padding will be filled with zeros.
    constant: int
        The value used to fill the padding when "constant" mode is
        selected.
        '''
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


def gen_warp_field(shape, sigma=0.1, grid_size=3):
    '''Generate an spline warp field'''
    import SimpleITK as sitk
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


def apply_warp(x, warp_field, fill_mode='reflect',
               interpolator=None,
               fill_constant=0, rows_idx=1, cols_idx=2):
    '''Apply an spling warp field on an image'''
    import SimpleITK as sitk
    if interpolator is None:
        interpolator = sitk.sitkLinear
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
    pattern = [el for el in range(0, x.ndim) if el not in [rows_idx, cols_idx]]
    pattern += [rows_idx, cols_idx]
    inv_pattern = [pattern.index(el) for el in range(x.ndim)]
    x = x.transpose(pattern)  # batch, channel, ...
    x_shape = list(x.shape)
    x = x.reshape([-1] + x_shape[2:])  # *, r, c
    warp_filter = sitk.WarpImageFilter()
    warp_filter.SetInterpolator(interpolator)
    warp_filter.SetEdgePaddingValue(np.min(x).astype(np.double))
    for i in range(x.shape[0]):
        bc_pad = pad_image(x[i], pad_amount=pad, mode=fill_mode,
                           constant=fill_constant).T
        bc_f = sitk.GetImageFromArray(bc_pad)
        bc_f_warped = warp_filter.Execute(bc_f, warp_field_padded)
        bc_warped = sitk.GetArrayFromImage(bc_f_warped)
        x[i] = bc_warped[pad:-pad, pad:-pad].T
    x = x.reshape(x_shape)  # unsquash
    x = x.transpose(inv_pattern)
    return x


def random_transform(x, y=None,
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     cval_mask=0.,
                     horizontal_flip=0.,  # probability
                     vertical_flip=0.,  # probability
                     rescale=None,
                     spline_warp=False,
                     warp_sigma=0.1,
                     warp_grid_size=3,
                     crop_size=None,
                     crop_mode='random',
                     smart_crop_threshold=0.5,
                     smart_crop_search_step=10,
                     smart_crop_random_h_shift_range=0,
                     smart_crop_random_v_shift_range=0,
                     return_optical_flow=False,
                     nclasses=None,
                     gamma=0.,
                     gain=1.,
                     chan_idx=3,  # No batch yet: (s, 0, 1, c)
                     rows_idx=1,  # No batch yet: (s, 0, 1, c)
                     cols_idx=2,  # No batch yet: (s, 0, 1, c)
                     void_label=None,
                     prescale=1.0):
    '''Random Transform.

    A function to perform data augmentation of images and masks during
    the training  (on-the-fly). Based on [RandomTransform1]_.

    Parameters
    ----------
    x: array of floats
        An image.
    y: array of int
        An array with labels.
    rotation_range: int
        Degrees of rotation (0 to 180).
    width_shift_range: float
        The maximum amount the image can be shifted horizontally (in
        percentage).
    height_shift_range: float
        The maximum amount the image can be shifted vertically (in
        percentage).
    shear_range: float
        The shear intensity (shear angle in radians).
    zoom_range: float or list of floats
        The amout of zoom. If set to a scalar z, the zoom range will be
        randomly picked in the range [1-z, 1+z].
    channel_shift_range: float
        The shift range for each channel.
    fill_mode: string
        Some transformations can return pixels that are outside of the
        boundaries of the original image. The points outside the
        boundaries are filled according to the given mode (`constant`,
        `nearest`, `reflect` or `wrap`). Default: `nearest`.
    cval: int
        Value used to fill the points of the image outside the boundaries when
        fill_mode is `constant`. Default: 0.
    cval_mask: int
        Value used to fill the points of the mask outside the boundaries when
        fill_mode is `constant`. Default: 0.
    horizontal_flip: float
        The probability to randomly flip the images (and masks)
        horizontally. Default: 0.
    vertical_flip: bool
        The probability to randomly flip the images (and masks)
        vertically. Default: 0.
    rescale: float
        The rescaling factor. If None or 0, no rescaling is applied, otherwise
        the data is multiplied by the value provided (before applying
        any other transformation).
    spline_warp: bool
        Whether to apply spline warping.
    warp_sigma: float
        The sigma of the gaussians used for spline warping.
    warp_grid_size: int
        The grid size of the spline warping.
    crop_size: tuple
        The size of crop to be applied to images and masks (after any
        other transformation).
    crop_mode: string
        The crop strategy, could be either 'random' or 'smart'.
        The 'random' mode randomly places the crop in the image.
        The 'smart' mode centers the crop in one of the locations where
        the non-background masks are more present through time (in case
        of sequences) or in the image otherwise. To do so it looks for a
        label called 'background' or 'void' to retrieve the mask id or
        assumes the id of the background mask to be 0.
    smart_crop_threshold: float in [0, 1]
        The percentage of background requested in the cropped image. If
        it is not possible to satisfy this constraint the smart cropping
        procedure returns the best crop found.
    smart_crop_search_step: int
        The amount of displacement in terms of of pixels used to search
        the best crop.
    smart_crop_random_h_shift_range: int
        The maximum horizontal random shift, in pixels, for 'smart' cropping
    smart_crop_random_v_shift_range: int
        The maximum vertical random shift, in pixels, for 'smart' cropping
    return_optical_flow: bool
        If not False a dense optical flow will be concatenated to the
        end of the channel axis of the image. If True, angle and
        magnitude will be returned, if set to 'rbg' an RGB representation
        will be returned instead. Default: False.
    nclasses: int
        The number of classes of the dataset.
    gamma: float
        Controls gamma in Gamma correction.
    gain: float
        Controls gain in Gamma correction.
    chan_idx: int
        The index of the channel axis.
    rows_idx: int
        The index of the rows of the image.
    cols_idx: int
        The index of the cols of the image.
    void_label: int
        The index of the void label, if any. Used for padding.

    References
    ----------
    .. [RandomTransform1]
       https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    '''
    # Set this to a dir, if you want to save augmented images samples
    save_to_dir = None  # "./"

    if x.ndim != 4:
        raise RuntimeError('The x input of random transform should have '
                           '4 dimensions. Received %d instead.' % x.ndim)
    if y.ndim != 3:
        raise RuntimeError('The y input of random transform should have '
                           '3 dimensions. Received %d instead.' % x.ndim)
    if rescale:
        raise NotImplementedError()

    # Do not modify the original images
    x = x.copy()
    if y is not None and len(y) > 0:
        y = y[..., None]  # Add extra dim to y to simplify computation
        y = y.copy()

    # Prescale each image/mask in the batch
    if prescale != 1.0:
        import skimage.transform
        x = [skimage.transform.rescale(x_image, prescale,
                                       order=1,  # bilinear
                                       preserve_range=True) for x_image in x]
        x = np.stack(x, 0)
        if y is not None and len(y) > 0:
            y = [skimage.transform.rescale(y_image, prescale,
                                           order=0,  # Nearest-neighbor
                                           preserve_range=True)
                 for y_image in y]
            y = np.stack(y, 0)

    # listify zoom range
    if np.isscalar(zoom_range):
        if zoom_range > 1.:
            raise RuntimeError('Zoom range should be between 0 and 1. '
                               'Received: ', zoom_range)
        zoom_range = [1 - zoom_range, 1 - zoom_range]
    elif len(zoom_range) == 2:
        if any(el > 1. for el in zoom_range):
            raise RuntimeError('Zoom range should be between 0 and 1. '
                               'Received: ', zoom_range)
        zoom_range = [1-el for el in zoom_range]
    else:
        raise Exception('zoom_range should be a float or '
                        'a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    # Channel shift
    if channel_shift_range != 0:
        x = random_channel_shift(x, channel_shift_range, rows_idx, cols_idx,
                                 chan_idx)

    # Gamma correction
    if gamma > 0:
        scale = float(1)
        x = ((x / scale) ** gamma) * scale * gain

    # Affine transformations (zoom, rotation, shift, ..)
    if (rotation_range or height_shift_range or width_shift_range or
            shear_range or zoom_range != [1, 1]):

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
                            cval=cval, order=1, rows_idx=rows_idx,
                            cols_idx=cols_idx)
        if y is not None and len(y) > 0:
            y = apply_transform(y, transform_matrix, fill_mode=fill_mode,
                                cval=cval_mask, order=0, rows_idx=rows_idx,
                                cols_idx=cols_idx)

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
        import SimpleITK as sitk
        warp_field = gen_warp_field(shape=(x.shape[rows_idx],
                                           x.shape[cols_idx]),
                                    sigma=warp_sigma,
                                    grid_size=warp_grid_size)
        x = apply_warp(x, warp_field,
                       interpolator=sitk.sitkLinear,
                       fill_mode=fill_mode,
                       fill_constant=cval,
                       rows_idx=rows_idx, cols_idx=cols_idx)
        if y is not None and len(y) > 0:
            y = np.round(apply_warp(y, warp_field,
                                    interpolator=sitk.sitkNearestNeighbor,
                                    fill_mode=fill_mode,
                                    fill_constant=cval_mask,
                                    rows_idx=rows_idx, cols_idx=cols_idx))

    """ Smart Cropping
    When the crop is performed in 'smart' mode the image or the frames in a
    video sequence are cropped trying to satisfy the costraint on the
    percentage of background pixels in the crop. This allows to crop
    in the area of the image where the foreground is more concentrated (or not)
    and to extrapolate parts of a video sequence where the most of the
    motion happens.
    In the following heuristic the crop is performed starting by the
    computation of the foreground mask that contains the foreground
    pixels in the case of an image and the sum of the foregorund pixels
    over all the sequence in the case of the video. The crop is first
    centred in one of the point that has the maximum 'concentration of
    foreground', then if foreground/background constraint is not
    satisfied the heuristic searches for another crop center by moving
    'smart_crop_search_step' in the direction chosen randomly between
    the possible direction in the remaining quadrants of the image (with
    respect to the quadrant where the current center is placed). The
    heuristic terminates when the threshold constraint is satisfied or
    when the border of the image is reached. If it is not possible to satisfy
    the fg/bg constraint for tthe current image or video sequence,
    the heuristic return the best crop found before.
    """
    if crop_mode == 'smart':
        # Compute a (n-1)D fg/bg binary mask (with nD input)
        if y.shape[-1] == 3:
            foreground_mask = (np.sum(y, axis=-1) > 0).astype(int)
        elif y.shape[-1] == 1:
            foreground_mask = np.squeeze((y > 0).astype(int), axis=-1)
        else:
            foreground_mask = np.squeeze(
                (np.expand_dims(y, -1) > 0).astype(int), axis=-1)
        # Accumulate masks over time
        if len(foreground_mask.shape) != 2:
            foreground_mask = np.sum(foreground_mask, axis=0)
        # Get background concentration in the four quadrants of the mask
        quadrants = []
        m_height = foreground_mask.shape[0]
        m_width = foreground_mask.shape[1]
        # Select the four quadrants of mask
        quadrants.append(foreground_mask[0:m_height // 2,
                                         0:m_width // 2])

        quadrants.append(foreground_mask[0:m_height // 2,
                                         m_width // 2:m_width])
        quadrants.append(foreground_mask[m_height // 2:m_height,
                                         m_width // 2:m_width])
        quadrants.append(foreground_mask[m_height // 2:m_height,
                                         0:m_width // 2])
        # Compute the max num of background pixels for each quadrant
        background_concentrations = []
        for quadrant in quadrants:
            rows, cols = np.where(quadrant == 0)
            background_concentrations.append(len(zip(rows, cols)))
        max_background_quadrant = background_concentrations.index(
            np.max(background_concentrations))
        quadrant_mask = np.zeros(foreground_mask.shape, foreground_mask.dtype)
        if max_background_quadrant == 0:
            b_rows, b_cols = np.where(
                foreground_mask[0:m_height // 2, 0:m_width // 2] == 0)
            quadrant_mask[(b_rows, b_cols)] = 1
        elif max_background_quadrant == 1:
            b_rows, b_cols = np.where(foreground_mask[0:m_height // 2,
                                      m_width // 2:m_width] == 0)
            quadrant_mask[(b_rows, b_cols + m_width // 2)] = 1
        elif max_background_quadrant == 2:
            b_rows, b_cols = np.where(
                foreground_mask[m_height // 2:m_height,
                                m_width // 2:m_width] == 0)
            quadrant_mask[(b_rows + m_height // 2, b_cols + m_width // 2)] = 1
        else:
            b_rows, b_cols = np.where(
                foreground_mask[m_height // 2:m_height,
                                0:m_width // 2] == 0)
            quadrant_mask[(b_rows + m_height // 2, b_cols)] = 1
        max_mask_value = np.max(foreground_mask)
        max_rows, max_cols = np.where(foreground_mask == max_mask_value)

    # Crop
    # Expects axes with shape (..., 0, 1)
    # TODO: Add center crop
    if crop_size:
        # Reshape to (..., 0, 1)
        pattern = [el for el in range(x.ndim) if el != rows_idx and
                   el != cols_idx] + [rows_idx, cols_idx]
        inv_pattern = [pattern.index(el) for el in range(x.ndim)]
        x = x.transpose(pattern)

        crop = list(crop_size)
        pad = [0, 0]
        h, w = x.shape[-2:]

        # Compute amounts
        if crop[0] < h:
            # Do random crop
            if crop_mode == 'random':
                top = np.random.randint(h - crop[0])
            elif crop_mode == 'smart':
                crop_center_row = np.random.choice(max_rows)
                # Random vertical shift
                v_shift = 0
                if smart_crop_random_v_shift_range > 0:
                    v_shift = np.random.randint(
                        -smart_crop_random_v_shift_range,
                        smart_crop_random_v_shift_range)
                crop_center_row += v_shift
                top = max(0, crop_center_row - crop_size[0] // 2)
                top = min(top, h - crop[0])
        else:
            # Set pad and reset crop
            pad[0] = crop[0] - h
            top, crop[0] = 0, h
        if crop[1] < w:
            # Do random crop
            if crop_mode == 'random':
                left = np.random.randint(w - crop[1])
            elif crop_mode == 'smart':
                crop_center_col = np.random.choice(max_cols)
                # Random horizontal shift
                h_shift = 0
                if smart_crop_random_h_shift_range:
                    h_shift = np.random.randint(
                        -smart_crop_random_h_shift_range,
                        smart_crop_random_h_shift_range)
                crop_center_col += h_shift
                left = max(0, crop_center_col - crop_size[1] // 2)
                left = min(left, w - crop[1])
        else:
            # Set pad and reset crop
            pad[1] = crop[1] - w
            left, crop[1] = 0, w

        if crop_mode == 'smart':
            # Search for requested f/b threshold
            background_portion = np.sum(
                foreground_mask[top:top+crop[0], left:left+crop[1]] == 0)
            current_threshold = background_portion / np.float(crop[0] *
                                                              crop[1])
            best_top = top
            best_left = left
            if current_threshold < smart_crop_threshold:
                best_threshold_found = False
                # Select randomly one of the background pixels of the
                # selected quadrant mask
                b_rows, b_cols = np.where(quadrant_mask == 1)
                random_b_x = np.random.choice(b_cols)
                random_b_y = np.random.choice(b_rows)
                if crop_center_col - random_b_x > 0:
                    max_left = 0
                    search_direction_x = -1
                else:
                    max_left = w - crop[1]
                    search_direction_x = 1
                if crop_center_row - random_b_y > 0:
                    max_top = 0
                    search_direction_y = -1
                else:
                    max_top = h - crop[0]
                    search_direction_y = 1
                while not best_threshold_found:
                    top += search_direction_y * smart_crop_search_step
                    top = max(0, top)
                    top = min(top, h - crop[0])
                    left += search_direction_x * smart_crop_search_step
                    left = max(0, left)
                    left = min(left, w - crop[1])
                    # Search for requested f/b threshold
                    background_portion = np.sum(
                        foreground_mask[top:top+crop[0],
                                        left:left+crop[1]] == 0)
                    old_threshold = current_threshold
                    current_threshold = background_portion / np.float(crop[0] *
                                                                      crop[1])
                    if current_threshold > old_threshold:
                        best_top = top
                        best_left = left
                    if current_threshold >= smart_crop_threshold:
                        best_threshold_found = True
                    elif top == max_top and left == max_left:
                        best_threshold_found = True
                        top = best_top
                        left = best_left

        # Cropping
        x = x[..., top:top+crop[0], left:left+crop[1]]
        if y is not None and len(y) > 0:
            y = y.transpose(pattern)
            y = y[..., top:top+crop[0], left:left+crop[1]]
        # Padding
        if pad != [0, 0]:
            pad_pattern = ((0, 0),) * (x.ndim - 2) + (
                (pad[0]//2, pad[0] - pad[0]//2),
                (pad[1]//2, pad[1] - pad[1]//2))
            x = np.pad(x, pad_pattern, 'constant')
            try:
                y = np.pad(y, pad_pattern, 'constant',
                           constant_values=void_label)
            except ValueError as e:
                raise type(e)(e.message + '\nCannot pad the image: the '
                              'dataset has no void class')

        x = x.transpose(inv_pattern)
        if y is not None and len(y) > 0:
            y = y.transpose(inv_pattern)

    if return_optical_flow:
        flow = optical_flow(x, rows_idx, cols_idx, chan_idx,
                            return_rgb=return_optical_flow == 'rgb')
        x = np.concatenate((x, flow), axis=chan_idx)

    # Save augmented images
    if save_to_dir:
        import seaborn as sns
        fname = 'data_augm_{}.png'.format(np.random.randint(1e4))
        cmap = sns.hls_palette(nclasses)
        save_img2(x, y, os.path.join(save_to_dir, fname),
                  cmap, void_label, rows_idx, cols_idx, chan_idx)

    # Undo extra dim
    if y is not None and len(y) > 0:
        y = y[..., 0]

    return x, y
