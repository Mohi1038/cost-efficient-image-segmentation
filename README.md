# CSL7360: Computer Vision — Assignment 2
## Cost-Efficient Saliency Segmentation under a Compute Budget

**Student Name / Roll Number**: Mohit / M24CSE018  
**Institution**: Indian Institute of Technology Jodhpur  
**Track Chosen**: Track B (Custom Small CNN — `FastSaliencyUNet`)

---

## 1. Executive Summary & Final Results

This project implements an ultra-compact, high-speed deep convolutional segmentation network designed from first principles to optimize the **Cost-Efficient Segmentation (CES)** score under strict hardware budgets:
- **CPU Time Budget**: Single-threaded execution $\le 50.0\text{ ms}$ on a single core per photo.
- **Disk Storage Budget**: Model parameter file $\le 4.194304\text{ MB}$ ($4\text{ MiB}$).

### Benchmark Summary Table

| Dataset / Split | Number of Images | mIoU | BF1 | Quality ($Q$) | $T_{CPU}$ (ms) | $S_{disk}$ | Budget Penalty | CES Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Set (DUTS-TR 10% Split)** | 1,055 | **0.8225** | **0.5830** | **0.7506** | 22.84 ms | 300,819 Bytes | 0.00% | **0.7506** |
| **DUTS-TE (Official Test Set)** | 5,019 | **0.6566** | **0.3704** | **0.5707** | **23.09 ms** | **300,819 Bytes** | **0.00%** | **0.5707** |
| **Captured Campus Photos** | 7 | **0.9583** | **0.9959** | **0.9696** | **22.35 ms** | 300,819 Bytes | 0.00% | **0.9696** |

> **Key Takeaway**: Because our single-core CPU time $T_{CPU} = 23.09\text{ ms} \le 50\text{ ms}$ and model weight file $S_{disk} = 300,819\text{ Bytes} \approx 0.287\text{ MB} \ll 4.0\text{ MB}$, both compute and storage penalties evaluate strictly to **zero**. The pipeline retains **100% of the Quality score**: $\text{CES} = \mathbf{0.5707}$.

---

## 2. Environment & Dependencies

- **Python Version**: Python 3.11+
- **Hardware Architecture**: x86_64 or Apple Silicon (arm64)

### Exact Package Versions
```text
torch==2.15.0
torchvision==0.29.2
opencv-python==5.0.0.93
numpy==2.5.4
pillow==13.0.0
matplotlib==3.11.2
tqdm==4.70.1
markdown==3.11.2
```

### Installation
```bash
pip install torch torchvision opencv-python numpy pillow matplotlib tqdm markdown
```

---

## 3. How to Run `predict.py`

`predict.py` takes a photo from disk, performs the full segmentation pipeline under single-thread CPU constraints, and outputs a binary mask PNG (0 for background, 255 for main object).

```bash
python predict.py --image path/to/input.jpg --out path/to/output_mask.png
```
*(Aliases `--input` and `--output` are also fully supported).*

### Example Output
```text
T CPU ms: 22.84
S disk bytes: 300819
```
The script prints exactly two lines:
1. `T CPU ms: <number>`: Average single-threaded CPU runtime over 20 repeated runs on the same photo.
2. `S disk bytes: <number>`: Size in bytes of `parameters.pth`.

---

## 4. How to Run Evaluation

To evaluate on DUTS validation split, test set, and captured photos:
```bash
# Full evaluation with single-thread CPU benchmark
python evaluate.py

# Fast evaluation skipping CPU timing
python evaluate.py --skip-timing
```

---

## 5. Repository Structure

```text
├── parameters.pth               # 300,819 Byte float16 quantized model parameters
├── predict.py                   # Required standalone inference script
├── model.py                     # FastSaliencyUNet architecture definition
├── dataset.py                   # DUTS and captured dataset loaders with augmentations
├── metrics.py                   # mIoU, Boundary F1 (BF1), and Quality Q calculations
├── evaluate.py                  # Evaluation script across validation, DUTS-TE, and captured sets
├── generate_report_figures.py   # Visual comparison and overlay figure generator
├── build_html_report.py         # Markdown-to-HTML report compiler
├── captured/                    # 7 phone photos covering all 5 required conditions
│   ├── original/                # Original full-resolution phone photos
│   ├── images_256/              # Resized 256x256 test photos
│   └── masks_256/               # Manually annotated ground truth masks (DUTS format)
├── report/
│   ├── REPORT.md                # Full academic project report (Markdown)
│   ├── REPORT.html              # Styled standalone HTML report
│   ├── REPORT.pdf               # Print-ready compiled PDF report
│   └── figures/                 # All generated high-resolution overlays and curves
└── README.md                    # Project documentation and reproduction guide
```

---

## 6. Captured Phone Photos

The repository includes 7 phone photos taken around the IIT Jodhpur campus and hostel environments, fulfilling all 5 mandated conditions:
1. **Indoor vs Outdoor**: `bag`, `bottle`, `shoes`, `watch`, `cycle` (indoor dorms & corridors) vs `cats`, `flower` (campus grounds).
2. **Daylight vs Tube Light**: `cats`, `flower` (natural sun) vs `bottle`, `bag`, `shoes`, `watch`, `cycle` (tube light & corridor fluorescents).
3. **Near vs Far Objects**: `watch`, `flower` (macro/close-up) vs `bottle`, `shoes`, `bag` (tabletop/floor) vs `cycle`, `cats` (scene scale).
4. **Cluttered Background**: `bottle` (study desk clutter), `bag` (patterned bedsheet), `watch` (swirled fabric), `cycle` (corridor floor & wall scuffs).
5. **Partial Occlusion & Complex Geometry**: `cats` (background cat partially occluded), `cycle` (fine spoke structure and frame geometry), `flower` (handheld petal grips).
