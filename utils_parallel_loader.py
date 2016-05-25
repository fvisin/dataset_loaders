import os
from skimage import io


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
    im_path = os.path.join(path, 'Original')
    if extension == "tiff":
        filename = str(video_index) + "_0.tiff"
    if extension == "jpg":
        filename = str(video_index) + "_0.jpg"
    img = io.imread(os.path.join(im_path, filename))

    return img.shape[0:2]
