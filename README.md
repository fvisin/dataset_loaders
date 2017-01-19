This repository contains some loaders for commonly used datasets. The loaders
can perform some on-the-fly preprocessing/data augmentation, as well as
run on multiple threads (if enabled) to speed up the I/O operations.

### How to install it:
1. Clone the repository with `--recursive` in some path, e.g. to your `$HOME`:

   ```sh
   git clone --recursive https://github.com/fvisin/dataset_loaders.git "$HOME/dataset_loaders"
   ```

2. Add that path to your `$PYTHONPATH` (replace `$HOME/dataset_loaders` with
   the path you cloned into):

   ```sh
   echo 'export PYTHONPATH=$PYTHONPATH:$HOME/dataset_loaders' >> ~/.bashrc
   ```

3. The framework assumes that the datasets are stored in some *shared paths*,
   accessible by everyone, and should be copied locally on the machines that
   run the experiments. The framework automatically takes care for you to copy
   the datasets from the *shared paths* to a *local path*. 
   
   Create a configuration file with these paths in 
   `/dataset_loaders/dataset_loaders/config.ini` (see the 
   [config.ini.example](dataset_loaders/config.ini.example) in the same 
   directory for guidance).
   
   Note: if you want to disable the copy mechanism, just specify the same path 
   for the local and the shared path:
   
   ```ini
   [general]
   datasets_local_path = /a/local/path
   [camvid]
   shared_path = /a/local/path/camvid
   [cityscapes]
   shared_path = /a/local/path/cityscapes/
   (and so on...)
   ```


4. To use the MS COCO dataset, you also need to do the following:

   ```sh
   cd dataset_loaders/images/coco/PythonAPI
   make all
   ```
4. You will need to install SimpleITK if you intend to use the *warp_spline*
   data augmentation:
   
   ```sh
    pip install SimpleITK --user  
   ```
</br>

**Note**: This framework is provided for research purposes only. The code
might be unstable. Please carefully check that it loads what you expect
them to load before using it!
 
</br>
</br>
</br>

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
