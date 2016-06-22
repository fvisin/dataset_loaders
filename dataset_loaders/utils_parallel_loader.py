import os
import re


def get_video_size(path):
    """
    Return the lengths of each video.

    :path: path to find the data_size.txt file
    """
    f = open(os.path.join(path, "data_size.txt"), "r")
    video_size = []
    for line in f:
        video_size.append(int(line))

    return len(video_size), video_size


def get_frame_size(path, video_index, extension="tiff"):
    """
    Find height and width of frames from one video.

    :path: path of the dataset
    :video_index: index of the video
    """
    from skimage import io
    im_path = os.path.join(path, 'Original')
    if extension == "tiff":
        filename = str(video_index) + "_0.tiff"
    if extension == "jpg":
        filename = str(video_index) + "_0.jpg"
    img = io.imread(os.path.join(im_path, filename))

    return img.shape[0:2]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]
