# Jetson Orin Setup Guide

## GPU Monitoring
```bash
sudo tegrastats | awk -F'GR3D_FREQ ' '{if (NF>1) {split($2,a,"%"); print "GPU:", a[1]"%"}}'
```

## Common Issues & Fixes

### CUDA Not Available
- Use NVIDIA-provided PyTorch wheels only
- install ```sudo apt install python3.8-venv```
- Create venv with: `python3 -m venv venv38 --system-site-packages`
- Never use `uv`, `conda`, or `pipx`

### PyTorch Installation
- Never run `pip install torch`
- Check JetPack version: `cat /etc/nv_tegra_release`
- Install matching NVIDIA wheel for your JetPack version

### Package Dependencies
- Remove all `nvidia-*` packages from requirements
- Pin `tokenizers==0.15.2`
- Pin `huggingface-hub==0.24.7`
- Disable HuggingFace Xet: `export HF_HUB_DISABLE_XET=1`
- Remove `pycolmap` (no ARM wheels)

### Environment Setup
- Use Python 3.8+
- Use `--system-site-packages` (warnings are expected)
- Upgrade pip if needed: `python -m pip install --upgrade pip`

## Key Rules
- PyTorch is a platform dependency, not a Python package
- Jetson uses system CUDA (JetPack) only
- venv is an overlay, not a sandbox


## Other issue
### ROS bridge issue
- ```attrs``` package version conflict
- uninstall using ```pip uninstall -y attr```
- execute without venv (in system python)
```bash
sudo apt update
sudo apt install --reinstall python3-attr python3-twisted
```
- verify with:
```bash
python3 - <<EOF
import attr
print("attr.s exists:", hasattr(attr, "s"))
EOF
```