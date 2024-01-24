# Adaptive Euclidean alignment
## Environment
* python 3.7 for deep learning
* python 2.7 for the PhysioNet Dataset （physionet_loadedf.py）
* pytorch 1.8.0
* braindecode package is directly copied from https://github.com/robintibor/braindecode/tree/master/braindecode for preparing datasets 
## Start
* setp 1 Prepare dataset(Only needs to run once) ps. physionet_loadedf.py requires python2.7, and this file may yield incorrect results when running under python3. x
   
    `python data/physionet_loadedf.py`
* setp 2 Process dataset(Only needs to run once)
  
    `python data/physionet_process.py`
* step 3 Train a pre-trained model 
  
    `python main_pretrain.py -data_path data/preprocess/physionet -id 11 -father_path save/physionet`
* step 4 Train a AEA-based SFDA model
  
    `python main_AEAGSFDA.py -data_path data/preprocess/physionet -id 11 -resume_path save/physionet/11`
## Licence
For academtic and non-commercial usage
