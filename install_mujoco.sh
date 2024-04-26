wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
ls ~/.mujoco/mujoco210/
vim ~/.bashrc
# export LD_LIBRARY_PATH=/home/xieming/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export PATH="$LD_LIBRARY_PATH:$PATH"
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
sudo apt-get update
sudo apt-get install patchelf
sudo apt-get install build-essential libssl-dev libffi-dev libxml2-dev
sudo apt-get install libxslt1-dev zlib1g-dev libglew1.5 libglew-dev 
ls /usr/lib/x86_64-linux-gnu/libGLEW.so
git clone git@github.com:openai/mujoco-py.git
cd mujoco-py
conda activate aibook
pip install -r requirements.txt
pip install -r requirements.dev.txt
pip3 install -e . --no-cache
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
ls -l /usr/lib/x86_64-linux-gnu/libGL.so
pip3 install -U 'mujoco-py<2.2,>=2.1'
pip install gymnasium[mujoco]