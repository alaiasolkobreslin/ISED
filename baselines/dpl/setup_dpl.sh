echo "Installing packages..."
conda install cython
pip install git+https://github.com/wannesm/PySDD.git#egg=PySDD
pip install problog
pip install deepproblog
echo "Setup complete."