This repository contains some loaders for commonly used datasets.

### How to use it:
1. Clone the repository with `--recursive`, e.g.:

   ```
   git clone --recursive https://github.com/fvisin/dataset_loaders.git
   ```
2. Add the path you cloned it into to the PYTHONPATH, e.g.:
   ```
   echo 'export PYTHONPATH=$PYTHONPATH:$HOME/dataset_loaders`' >> ~/.bashrc
   ```
   (change `$HOME/dataset_loaders` to whatever path you cloned it into)
3. In the inner `dataset_loaders` directory (e.g. `$HOME/dataset_loaders/dataset_loaders`) 
   create a symbolic link named `datasets` to wherever you have the datasets:
   ```
   ln -s /path/to/datasets/ $HOME/dataset_loaders/dataset_loaders/datasets
   ```
   
   Be sure to use full paths (that start from `/` or `$HOME`) or the symbolic link won't
   work.
4. To use the MS COCO dataset, you also need to do the following:
   ```
   cd dataset_loaders/images/coco/PythonAPI
   make all
   ```
