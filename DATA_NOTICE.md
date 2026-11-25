# Data, copyright and third-party licenses

The MIT license in [LICENSE](LICENSE) applies to the **code** in this repository only.

## The VietMuong dataset is not in this repository

The data are distributed separately on the Hugging Face Hub:
<https://huggingface.co/datasets/LKAusagi/VietMuong>. Access requires accepting the
terms below on that page. The training script downloads the data from there
(pinned revision `d129079b0d8bf2c887f41ad879d884d9f3ad15c0`).

## Source of the data

VietMuong was extracted from a printed bilingual dictionary and manually verified
against it:

> V. K. Nguyễn, C. Bùi and V. H. Hoàng. *Từ điển Mường–Việt* [Muong–Vietnamese
> Dictionary]. Hà Nội: Nhà xuất bản Văn hóa dân tộc, 2002.

The dictionary is a copyrighted work. Copyright in its content, including its example
sentences and the selection and arrangement of its entries, remains with its authors
and publisher. The VietMuong authors do not claim copyright over that content and do
not grant any rights to it.

## Terms of use for the data

- The dataset is provided for **non-commercial research and education** on
  Vietnamese–Muong language technology.
- Credit the paper (citation details in the [README](README.md#citation) once
  published) and the source dictionary above when you use the data.
- Do not redistribute the data; point others to the dataset page, where access
  requires accepting these terms.
- Do not use the data to reconstruct, republish or sell the dictionary or a
  substitute for it.
- Rights holders who object to the distribution of any part of the data can contact
  the authors (lekhaank16@siu.edu.vn); the affected content will be removed.

## Files that must not be committed here

- scans, page images or OCR output of the dictionary;
- the CSV files of the dataset itself (use the Hub);
- prediction files (`predictions/*.csv`), because they contain the source and
  reference sentences of the test sets.

## Third-party models and tools

Fine-tuned checkpoints are not distributed. Anyone who fine-tunes or shares models
must also follow the licenses of the pre-trained models used in the paper:

| Model / tool | License |
|---|---|
| ViT5-base, ViT5-large (`VietAI/vit5-*`) | MIT |
| mT5-base, mT5-large (`google/mt5-*`) | Apache-2.0 |
| ByT5-small, ByT5-large (`google/byt5-*`) | Apache-2.0 |
| NLLB-200 distilled 600M / 1.3B (`facebook/nllb-200-distilled-*`) | CC-BY-NC-4.0 (non-commercial) |
| Qwen3-VL-2B-Instruct (transcription, Sec. III-A) | Apache-2.0 |
| EasyOCR (text detection, Sec. III-A) | Apache-2.0 |

Models fine-tuned from NLLB-200 inherit its non-commercial restriction.
