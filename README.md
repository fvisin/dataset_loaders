This repository contains some loaders for commonly used datasets.

### How to use it:
1. Clone the repository *somewhere*.
2. Add *somewhere* to the PYTHONPATH, i.e. in Linux add to `~/.bashrc` 
   `export PYTHONPATH=$PYTHONPATH:/path/to/somewhere`.
3. Create a symbolic link named `datasets` to wherever you have the datasets:
   `ln -s /path/to/datasets/ /path/to/somewhere/dataset_loaders/datasets`
   Be sure to use a full path (from `/`) or the symbolic link won't work.
