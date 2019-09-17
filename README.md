# RIPR
Resolution-invariant Person Re-identification
accepted by IJCAI2019
https://arxiv.org/abs/1906.09748

This code is mainly encouraged by https://github.com/KaiyangZhou/deep-person-reid

To accelerate evaluation (10x faster), you can use cython-based evaluation code (developed by [luzai](https://github.com/luzai)). First `cd` to `torchreid/eval_lib`, then do `make` or `python setup.py build_ext -i`. After that, run `python test_cython_eval.py` to test if the package is successfully installed.

dataset:

  VR dataset, which can be constructed by downsampling the original dataset. If you want to download the dataset please contact snmao@pku.edu.cn . And put the dataset in folder data/ For example, data/vr_market1501/query/XXX

train:

python RIPR.py -d market1501 --optim adam --lr 0.0003 --max-epoch 60 --stepsize 20 40 --train-batch 32 --test-batch 100 --save-dir log/RIPR_train --gpu-devices 0

test:

python RIPR.py --evaluate -d market1501 --test-batch 100 --save-dir log/RIPR_train --gpu-devices 0 --testmodel log/RIPR_train/best_model.pth.tar

model:

the model file can be downloaded from https://pan.baidu.com/s/1aOBPaiFnlS7BRJWnXvaOew  , whose password is yov0



