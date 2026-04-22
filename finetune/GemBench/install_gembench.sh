FINETUNE_ROOT=/robot/robot-research-exp-0/user/lpy/BridgeVLA_sam/finetune

pip install wheel ninja pyyaml
cd $FINETUNE_ROOT
#wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
#tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
export COPPELIASIM_ROOT=$FINETUNE_ROOT/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:1.0

pip3 install pip==25.0.1
pip3 install setuptools==76.1.0
pip3 install open3d
pip install -e .
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install torchaudio==2.5.1 torchvision==0.20.1
pip install 'accelerate>=0.26.0'
pip3 install transformers==4.51.3  # note that 4.52.3 provide different model keys for paligemma
pip install git+https://github.com/openai/CLIP.git
apt-get update
apt-get install -y libffi-dev
apt-get install -y xvfb
apt-get install -y libfontconfig1
apt install -y libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0
apt install -y libxcb-cursor0
apt install -y libxcb-xinerama0
pip install pyqt6
pip3 install yacs
pip3 install swanlab
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'

apt-get update
apt-get install -y libxcb-xinput0 libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0 libxcb-xfixes0 libxcb-shape0 libxcb-randr0 libxcb-image0 libxcb-keysyms1 libxcb-icccm4 libxcb-sync1 libxcb-xinerama0 libxcb-util1
apt-get install -y libxcb-glx0 libxcb-xkb1 libxkbcommon-x11-0
apt install -y ffmpeg
pip install ffmpeg-python


cd $FINETUNE_ROOT/bridgevla/libs
git clone https://github.com/cshizhe/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install .
cd ..

#  modified version of RLBench to support new tasks in GemBench
git clone https://github.com/rjgpinel/RLBench
cd RLBench
pip install -r requirements.txt
pip install .
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless

pip install -e $FINETUNE_ROOT/bridgevla/libs/YARR
pip install -e $FINETUNE_ROOT/bridgevla/libs/peract_colab
pip install -e $FINETUNE_ROOT/bridgevla/libs/point-renderer


cd $FINETUNE_ROOT/GemBench
pip install -e .
pip3 install msgpack_numpy
pip3 install jsonlines
pip3 install lmdb