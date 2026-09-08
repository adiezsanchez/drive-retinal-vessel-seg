# Data

This repository **does not redistribute** the official DRIVE photographs.

## Synthetic demo (always available)

```bash
pixi run generate-data
```

writes a DRIVE-like folder tree under `data/synthetic/`:

```
data/synthetic/
  training/{images,1st_manual,mask}/
  test/{images,1st_manual,mask}/
```

These images are procedural (optic disc + branching vessels + circular FOV).
They exist so every `pixi run` task and notebook produces real figures without
a Grand Challenge account. They are **not** clinical data.

## Official DRIVE

1. Register at [drive.grand-challenge.org](https://drive.grand-challenge.org/).
2. Download the training/test archives from
   [the official Download page](https://drive.grand-challenge.org/Download/).
3. Extract:

   ```bash
   pixi run download-drive -- --archive /path/to/DRIVE.zip
   ```

The loader in `src/drive_seg/data.py` **prefers** `data/drive/` when that
layout is present, and otherwise falls back to `data/synthetic/`.

Typical official layout (Staal et al., IEEE TMI 2004):

```
data/drive/
  training/
    images/          # 21_training.tif … 40_training.tif
    1st_manual/      # 21_manual1.gif …
    mask/            # 21_training_mask.gif …
  test/
    images/          # 01_test.tif … 20_test.tif
    1st_manual/      # present in many distributions
    mask/
```

Images are 584×565 RGB fundus photographs; vessels are annotated inside a
circular field of view of about 540 px diameter.
