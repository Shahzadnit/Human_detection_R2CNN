## Installation



### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name maskrcnn
conda activate maskrcnn

clone the repository

go to directory  
# maskrcnn_benchmark and coco api dependencies
pip install -r requirements.txt

python setup.py build develop

```

