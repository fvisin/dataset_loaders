.. _index_txt:

:tocdepth: 4

Welcome to the Dataset loaders' documentation!
==============================================

This repository contains a framework to load the most commonly used datasets
for image and video semantic segmentation. The framework can perform some
on-the-fly preprocessing/data augmentation, as well as run on multiple threads
(if enabled) to speed up the I/O operations.

.. seealso::

    **NEWS!** You might be interested in checking out `Main loop TF <https://github.com/fvisin/main_loop_tf/>`_, 
    a python main loop that integrates the Dataset loaders with Tensorflow!

Attribution
===========

.. warning::

    If you use this code, please cite:

        Francesco Visin, Adriana Romero, (2016). *Dataset loaders: a python
        library to load and preprocess datasets* `[BibTex]
        <https://gist.github.com/fvisin/7104500ae8b33c3b65798d5d2707ce6c#file-dataset_loaders-bib/>`_

Quick start
===========

1. Clone the repository with `--recursive` in some path, e.g. to your `$HOME`::

       git clone --recursive https://github.com/fvisin/dataset_loaders.git "$HOME/dataset_loaders"

2. Install the package::

       pip install [--user] -e <dataset_loaders path> -r requirements.txt

3. The framework assumes that the datasets are stored in some *shared paths*,
   accessible by everyone, and should be copied locally on the machines that
   run the experiments. The framework automatically takes care for you to copy
   the datasets from the shared paths to a *local path*. 

   Create a configuration file in 
   ``[..]/dataset_loaders/dataset_loaders/config.ini`` to specify these paths
   (see the `config.ini example <https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/config.ini.example>`_
   in the same directory for guidance).

   Note: if you want to disable the copy mechanism, just specify the same path 
   for the local and the shared path::

       [general]
       datasets_local_path = /a/local/path
       [camvid]
       shared_path = /a/local/path/camvid
       [cityscapes]
       shared_path = /a/local/path/cityscapes/

       (etc...)

4. To use the MS COCO dataset, you also need to do the following::

       cd dataset_loaders/images/coco/PythonAPI
       make all

4. You will need to install
   `SimpleITK <https://simpleitk.readthedocs.io/en/master/index.html>`_
   and `openCV <http://opencv.org/>`_ if you intend to use the *warp_spline* or
   the *optical flow* data augmentations respectively.



Disclaimer
==========
**The code is provided as is, please expect minimal-to-none support on it.**

This framework is provided for research purposes only. Although we tried our 
best to test it, the code might be bugged or unstable. Use it at your own
risk!

The framework currently supports image or video based datasets. It could be 
easily extended to support other kinds of data (e.g., text corpora), but
there is no plan on our side to work on that at the moment.
* Feel free to contribute to the code with a PR if you find bugs, want to
improve the existing code or add support for other datasets.

Licence: `GNU General Public License v3.0 
<https://github.com/fvisin/dataset_loaders/blob/master/LICENSE.txt>`_.
 

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.


Index
=====

* :ref:`genindex` - Alphabetical index of content
* :ref:`modindex` - List of modules
