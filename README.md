This repository contains some loaders for commonly used datasets. The loaders can perform some on-the-fly preprocessing/data augmentation, as well as potentially run on multiple threads to speed up the I/O operations.

The code assumes that the datasets are stored in a **shared path**, accessible by everyone, and will be copied **locally** on the machines that run experiments.

### How to install it:
1. Clone the repository with `--recursive` in some path, e.g. to your `$HOME`:

   ```
   git clone --recursive https://github.com/fvisin/dataset_loaders.git "$HOME/dataset_loaders"
   ```
   
2. Add that path to your PYTHONPATH (replace `$HOME/dataset_loaders` with the path you cloned into):

   ```
   echo 'export PYTHONPATH=$PYTHONPATH:$HOME/dataset_loaders' >> ~/.bashrc
   ```
   
3. In the inner `dataset_loaders` directory (e.g. `$HOME/dataset_loaders/dataset_loaders`) create 
   a symbolic link (or a directory) named `dataset`. This directory will be used to save a **local copy** 
   of the datasets:

   ```
   ln -s /path/to/datasets/ "$HOME/dataset_loaders/dataset_loaders/datasets"
   ```
   
   NOTE: use full paths (that start from `/` or `$HOME`) for symbolic links or they won't work.
4. Edit the `config.ini` file (in the inner `dataset_loaders` directory). For each dataset specify the 
   **shared path** where the original files can be found.
4. To use the MS COCO dataset, you also need to do the following:

   ```
   cd dataset_loaders/images/coco/PythonAPI
   make all
   ```
4. You will need to install SimpleITK for data augmentation:
   ```
    pip install SimpleITK --user  
   ```

**Note**: These loaders might still be unstable, please carefully check that they are loading what you expect them to load before using them!
