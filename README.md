This repository contains some loaders for commonly used datasets.

### How to use it:
1. Clone the repository *somewhere*, e.g. `~/dataset_loaders`
2. Add it to the PYTHONPATH, i.e. in Linux add something like this to your
   `~/.bashrc`: `export PYTHONPATH=$PYTHONPATH:$HOME/dataset_loaders`.
3. In the `dataset_loaders` directory inside the repository (the inner one) 
   create a symbolic link named `datasets` to wherever you have the datasets:
   `ln -s /path/to/datasets/ $HOME/dataset_loaders/dataset_loaders/datasets`
   Be sure to use a full path (from `/` or `$HOME`) or the symbolic link won't
   work.
4. To use the MS COCO dataset, you also need to do the following:

    cd dataset_loaders/images/coco/PythonAPI
    make all
