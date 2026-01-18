Here is **everything in ONE single `.md` file**, clean and ready to save as e.g.
`JETSON_SETUP_NOTES.md` 👇

````md
# Jetson Orin – Python, CUDA & ML Stack Issues (Summary & Fixes)

This document summarizes all major issues encountered while setting up
PyTorch + CUDA + a large ML Python stack on **Jetson Orin (JetPack 5.1.3)**,
and how each issue was resolved.

---

## 1. `torch.cuda.is_available() == False`

**Cause**
- Installed PyTorch from PyPI / uv / conda
- Wrong Python interpreter (Python 2)
- venv isolated system CUDA libraries

**Fix**
- Use **NVIDIA-provided PyTorch wheel only**
- Use **Python 3.8**
- Create venv with system packages:
  ```bash
  python3 -m venv venv38 --system-site-packages
````

* Never run `pip install torch`

---

## 2. `uv`, conda, or isolated venv broke CUDA

**Cause**

* These tools hide Jetson’s system CUDA libraries

**Fix**

* Do **NOT** use:

  * `uv`
  * `conda`
  * `pipx`
* Use system Python + `venv --system-site-packages`

---

## 3. Wrong PyTorch wheel / JetPack mismatch

**Cause**

* Using wrong JetPack folder (`v511`, `v512`, etc.)

**Fix**

* Check JetPack:

  ```bash
  cat /etc/nv_tegra_release
  ```
* Map L4T → JetPack → wheel folder
  Example (JetPack 5.1.3):

  ```text
  v513
  ```
* Install matching NVIDIA wheel.

---

## 4. Installing `nvidia-*` packages via pip

**Cause**

* Desktop CUDA wheels installed on Jetson

**Fix**

* Remove **ALL** `nvidia-*` entries from requirements
* Jetson uses **system CUDA only (JetPack)**

---

## 5. `puccinialin` error (hf-xet / HF backend)

**Symptoms**

* Errors during:

  * `hf-xet`
  * `huggingface-hub`
  * `tokenizers`
* Message:

  ```text
  No matching distribution found for puccinialin
  ```

**Cause**

* HuggingFace Xet backend auto-enabled on ARM
* Broken build backend dependency

**Fix**

* Disable Xet globally:

  ```bash
  export HF_HUB_DISABLE_XET=1
  ```
* (Optional, cleaner) Pin:

  ```txt
  huggingface-hub==0.24.7
  ```

---

## 6. `tokenizers` build failure on ARM

**Cause**

* Newer `tokenizers` versions build from source
* Triggers broken HF backend on Jetson

**Fix**

* Force prebuilt ARM wheel:

  ```txt
  tokenizers==0.15.2
  ```

---

## 7. `pycolmap` not installable

**Cause**

* No ARM / aarch64 wheels on PyPI

**Fix**

* Remove `pycolmap` from Jetson requirements
* Build from source **only if absolutely necessary**

---

## 8. `pip cache purge` not available

**Cause**

* Old pip version (common on Ubuntu / Jetson)

**Fix**

* Either upgrade pip:

  ```bash
  python -m pip install --upgrade pip
  ```
* Or manually clear cache:

  ```bash
  rm -rf ~/.cache/pip
  ```

---

## 9. “Not uninstalling … outside environment” warnings

**Cause**

* Using `--system-site-packages`

**Fix**

* Ignore — this is **expected and correct**
* System packages remain untouched

---

## Golden Rules for Jetson (Must Follow)

* PyTorch is a **platform dependency**, not a Python dependency
* Never put `torch` or `torchvision` in `requirements.txt`
* Never install `nvidia-*` CUDA packages via pip
* Pin `tokenizers` on ARM
* Disable HuggingFace Xet
* Use Python 3 only
* venv is an **overlay**, not a sandbox

---

## Final Result

✅ CUDA working
✅ NVIDIA PyTorch working
✅ Large ML stack installed
✅ Stable, reproducible Jetson Orin environment

This setup is **production-safe for Jetson Orin**.

```

If you want, I can also:
- Convert this into a **README.md**
- Add a **one-command setup script**
- Split **desktop vs Jetson** notes
- Add ROS-specific notes

Just tell me 👍
```
