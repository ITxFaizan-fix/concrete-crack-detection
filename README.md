# Concrete Crack Detection (CNN transfer learning)

Semester project, Artificial Intelligence course (P176B041). Author: Faizan Haider.

## What it does
Classifies a close-up photo of concrete as **crack / no crack** with MobileNetV2 transfer learning, and shows a Grad-CAM heat map of where the model looked. Informational tool only; it does not replace human inspection.

## Dataset
SDNET2018 (Utah State University, CC BY 4.0): 56,092 labelled images of concrete decks, pavements and walls, about 15% with cracks. DOI: https://doi.org/10.15142/T3TD19
Images are cut from 230 source photos, so the train/validation/test split is made **by source photo** (no photo appears in two sets): 40,305 / 7,992 / 7,795 images.

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Majority class | 0.823 | 0.000 | 0.000 | 0.000 |
| Random Forest (simple features) | 0.702 | 0.330 | 0.662 | 0.441 |
| MobileNetV2 transfer learning | 0.910 | 0.809 | 0.643 | 0.717 |

MobileNetV2 ROC AUC: 0.906. Decision thresholds were chosen on the validation set only.

## How to run
1. Open the notebook in Google Colab and set Runtime -> T4 GPU.
2. Download SDNET2018 from the DOI above and put `SDNET2018.zip` in your Google Drive.
3. Run the cells from top to bottom. The last cell starts the Gradio demo.
4. Install dependencies locally with `pip install -r requirements.txt`.

## Files
- notebook: full code (data split, baseline, MobileNetV2, Grad-CAM, demo)
- fig_*.png: training curves, confusion matrix, Grad-CAM examples
- baseline_test.json, cnn_test.json: test metrics

## Limitations
Some images contain very thin cracks, shadows and joints that are easy to confuse. The model misses about a third of cracks, so it must not be used alone for safety decisions.
