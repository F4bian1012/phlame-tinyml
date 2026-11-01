# Attribution & Licensing — `data/sample/`

This directory ships an 8-image smoke-test sample (**4 `non_person`** and **4 `person`**, preprocessed to `160×120` grayscale with EXIF stripped) drawn from the held-out `test` partition (`hil_subset == 1`) of the **Visual Wake Words (VWW)** benchmark subset (`data/splits_vww/test_manifest.csv`).

No personal data or vehicle license plates from the primary application dataset are included in this repository.

## Licenses

1. **Dataset Annotations (MS-COCO 2017):** Licensed by the COCO Consortium under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) (see [cocodataset.org/#termsofuse](https://cocodataset.org/#termsofuse)).
2. **Individual Photographs:** All 8 images below carry `license = 4` in `instances_val2017.json`, corresponding to the permissive **[Creative Commons Attribution 2.0 Generic License (CC BY 2.0)](https://creativecommons.org/licenses/by/2.0/)** (no NonCommercial or NoDerivatives restrictions). Modification applied: converted from RGB to grayscale and resized to `160×120` pixels (`src/process_images.py` + `src/reshape_images.py`).

## Per-Image Provenance & Checksums

| Class | File | COCO `image_id` | Photo License | Original Flickr URL | MS-COCO `val2017` URL | SHA-256 (`160×120` grayscale) |
|---|---|---:|---|---|---|---|
| `non_person` | `000000191580.jpg` | `191580` | CC BY 2.0 | http://farm5.staticflickr.com/4092/4987532314_ac508c6b1c_z.jpg | http://images.cocodataset.org/val2017/000000191580.jpg | `ed529f32007549017f0cdda92ad97486267d866a9cef792c775725af0ff74099` |
| `non_person` | `000000330790.jpg` | `330790` | CC BY 2.0 | http://farm8.staticflickr.com/7046/6949409387_e3d3ee659d_z.jpg | http://images.cocodataset.org/val2017/000000330790.jpg | `f146db14fe21b33541d33e5488ee2bc2384c61eba8d2de374831aa3c587c3701` |
| `non_person` | `000000343076.jpg` | `343076` | CC BY 2.0 | http://farm4.staticflickr.com/3528/3940136096_1831da02ea_z.jpg | http://images.cocodataset.org/val2017/000000343076.jpg | `da2cff72234b4df852927c89a760bad907f1cbf6f63e54333cb22d5ae1d20957` |
| `non_person` | `000000520871.jpg` | `520871` | CC BY 2.0 | http://farm3.staticflickr.com/2548/3711888643_b930d29ec8_z.jpg | http://images.cocodataset.org/val2017/000000520871.jpg | `0b642363916359db9133baacaef564b3e598cde1d08eba9bec84743e89c71e9a` |
| `person` | `000000000872.jpg` | `872` | CC BY 2.0 | http://farm9.staticflickr.com/8447/7805810128_605424213d_z.jpg | http://images.cocodataset.org/val2017/000000000872.jpg | `ec9a59d06c21deaeef37721f826b85f771c4cce381239be6ef06185de0696594` |
| `person` | `000000021903.jpg` | `21903` | CC BY 2.0 | http://farm9.staticflickr.com/8088/8456686192_fccee8c5bb_z.jpg | http://images.cocodataset.org/val2017/000000021903.jpg | `655be5e77a7bc944806c4c8df9efde86311da8fd221bd70869bbb1f4d359848c` |
| `person` | `000000064898.jpg` | `64898` | CC BY 2.0 | http://farm5.staticflickr.com/4094/4891067097_caa3cccb18_z.jpg | http://images.cocodataset.org/val2017/000000064898.jpg | `3e25e390b6c4dfa5f90d769109333e0e0941b5c5af6367c62eb6484fc1c82885` |
| `person` | `000000295316.jpg` | `295316` | CC BY 2.0 | http://farm8.staticflickr.com/7049/8688630144_370a671fba_z.jpg | http://images.cocodataset.org/val2017/000000295316.jpg | `fc2a8b0b95671c3ea605c52e148ee98c0044994337884c671e64814a1363c172` |

## References

- T.-Y. Lin *et al.*, "Microsoft COCO: Common Objects in Context," *ECCV*, 2014.
- A. Chowdhery, P. Warden, J. Shlens, A. Howard, and R. Rhodes, "Visual Wake Words Dataset," *arXiv:1906.05721*, 2019.
