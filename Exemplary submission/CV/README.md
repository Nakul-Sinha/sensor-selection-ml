# Exemplary Submission - Aneurysm Volume Prediction (Computer Vision)

Copied from: https://shipd.ai/quests/eris/docs/examples/aneurysm-volume-prediction

## Start here
1. `PAGE_CONTENT.md` - full copied material + embedded scripts
2. `01_dataset.md` / `02_problem.md` / `03_solution.md` - Dataset / Problem / Solution tab text (verbatim)
3. `aneurysm-volume-prediction.md` - formatted single-file version (prose + code, nicely structured)
4. `scripts/` - all code blocks from the page

## Scripts
- `scripts/prepare_script.py` - prepare/split script from Dataset tab
- `scripts/grading_script.py` - grading/metric script from Problem tab
- `scripts/config.yaml` - challenge config from Problem tab
- `scripts/solution_code.py` - exemplary end-to-end solution from Solution tab

## Binary files listed on the page (not downloadable - the docs pages only show file name/size cards, no public URLs)
- solution.ipynb (15 KB) - Solution tab output files
- working/unet3d_best.pt (1.2 MB) - Solution tab output files (trained model checkpoint)
- submission.csv (1 KB) - Solution tab output files
- Dataset files described in prose: public/train/ (49 .nii.gz patches), public/train_labels/ (49 masks), public/train.csv, public/test/ (10 patches), public/sample_submission.csv, private/answers.csv

Names, sizes, and descriptions are preserved verbatim in the Dataset/Solution text. All readable page content and every script/code block were copied into this folder. `dataset/` is present but empty for that reason.