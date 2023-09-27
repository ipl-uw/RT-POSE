pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
cd det3d/ops/dcn && python setup.py build_ext --inplace
cd ../../..