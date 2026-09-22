# ICG-VIMOD

**A dual-stream visible–infrared fusion object detector built on YOLO11.**

This repository is the official code release for the paper *ICG-VIMOD*
(see [Citation](#citation)). It is derived from
[Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) and extends it into a
two-stream detector that consumes **paired visible (VI) and infrared (IR) images** and
fuses them at the backbone and neck levels.

> **Status:** the dataset is distributed through GitHub Releases — see
> [Dataset](#dataset). More materials are being added, see
> [Upcoming updates](#upcoming-updates).

---


### What was changed compared to YOLO11

All modifications live inside the vendored `ultralytics/` package:

| File | Change |
| --- | --- |
| `ultralytics/cfg/models/11/yolo11fuse.yaml` | The dual-stream architecture: parallel VI and IR backbones, cross-modal fusion, detection head. |
| `ultralytics/nn/modules/block.py` | New modules: `getvi`, `getir`, `feature_fusion`, `ConcatFusion`, `InfraredEdgeModule`, `C3k2_irlb`, `C3k2_irat`. |
| `ultralytics/engine/trainer.py` | A second dataloader for the IR stream; the two batches are concatenated along the batch dimension (`torch.cat((batch['img'], batch_ir['img']), dim=0)`) and split again in the network by `getvi` / `getir`. |
| `ultralytics/engine/validator.py` | Paired validation over both modalities. |
| `ultralytics/cfg/datasets/data.yaml` | Dataset definition, including the custom `train_ir` / `val_ir` splits. |
| `train.py`, `val.py` | Entry points for training and validation. |

---

## Dataset

The dataset (`dataset_irvi`) is **not stored in this repository** — it is published as a
GitHub Release asset:

**Download:** <https://github.com/zzzzzzzcccc/icgvimod/releases/latest> → `dataset_irvi.zip`

**The released dataset is not split.** It is a single pool of 4,131 paired
visible/infrared images with no train/validation division — divide it into your own
train/val split as needed. Extract it so that it sits **next to this repository**, as in
our experiments:

```
E:\ICG-VIMOD\
├── ICG-VIMOD\              <- this repository
└── dataset_irvi\           <- extracted dataset
    ├── images\
    │   ├── vi\             (4,131 visible images)
    │   └── ir\             (4,131 infrared images, paired 1:1 with vi\)
    └── labels\
        ├── vi\
        └── ir\             (YOLO format: class cx cy w h)
```

Each image in `images/vi/` has a same-named counterpart in `images/ir/` (e.g. `0000.png`
in both); the two are fed to the network as a pair.

### Classes

| ID | Name | ID | Name |
| --- | --- | --- | --- |
| 0 | Soldering Iron | 4 | Heating Mantle |
| 1 | Hot Plate | 5 | Electric Furnace |
| 2 | Heat Gun | 6 | Hot cutter |
| 3 | Alcohol Lamp | 7 | Alcohol Blast Lamp |

### Pointing the code at your copy

`ultralytics/cfg/datasets/data.yaml` uses absolute paths for its `train` / `val` /
`train_ir` / `val_ir` keys. The released dataset ships unsplit, so create your own split
and edit those keys to point at it.

> **Note:** `train_ir` and `val_ir` are custom keys read directly by our trainer, which —
> unlike the standard `train` / `val` keys — does **not** resolve them against `path`.
> They must therefore be absolute.

---

Run `train.py` and `val.py` **from the repository root**, so that the local
`ultralytics/` package is imported instead of any pip-installed copy.

---

## Training

```bash
python train.py
```

The defaults in `train.py` reproduce our final run: `yolo11fuse.yaml`, 640 × 640, 1,000
epochs, batch 64. Change `device` to match your machine (it is set to `'1'`, i.e. the
second GPU).

```python
model = YOLO(model=r'ultralytics/cfg/models/11/yolo11fuse.yaml')
model.train(data=r'ultralytics/cfg/datasets/data.yaml',
            imgsz=640, epochs=1000, batch=64, workers=8, device='1', name='final')
```

## Validation

```bash
python val.py
```

```python
model = YOLO(r'runs/detect/final/weights/best.pt')
model.val(data=r'ultralytics/cfg/datasets/data.yaml', imgsz=640, batch=64, device=0)
```


## Upcoming updates

- Continue to expand and refine the dataset.

## Citation

> **TODO — fill in before publishing the repository.** The paper metadata below is a
> placeholder.

```bibtex
@article{icgvimod,
  title   = {ICG-VIMOD: <paper title>},
  author  = {<authors>},
  journal = {<journal>},
  year    = {<year>}
}
```

If you use this code, please also cite the Ultralytics YOLO project it builds on:

```bibtex
@software{yolo11_ultralytics,
  author  = {Glenn Jocher and Jing Qiu},
  title   = {Ultralytics YOLO11},
  year    = {2024},
  version = {11.0.0},
  url     = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

## License and acknowledgements

This project is a derivative work of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
and is therefore released under the **AGPL-3.0** license — see [LICENSE](LICENSE).

```bibtex
@software{yolov8_ultralytics,
  author  = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title   = {Ultralytics YOLO},
  year    = {2023},
  url     = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

We thank the Ultralytics team and the YOLO community for the base implementation.
