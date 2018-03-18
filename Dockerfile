FROM nvidia/cuda:8.0-cudnn6-runtime
RUN apt-get update
RUN apt-get -y install python3-dev python3-pip python3-tk curl
RUN pip3 install -U pip
RUN pip3 install tensorflow-gpu==1.4.1 tqdm imageio h5py
