# Stack Exchange question classifier

An R coursework project comparing text classifiers across four Stack Exchange question categories: world building, science fiction, English language and English learners. The original analysis uses 8,000 questions, balanced across the four categories.

## Pipeline

1. Combine title and body text and create a stratified 80/20 train/test split.
2. Clean text with `tm`, remove stop words and stem terms.
3. Build text features and compare LASSO multinomial regression, random forest and a Keras neural network.
4. Export model comparison tables, confusion matrices and per-class F1 results.

## Run

Use R with `tidyverse`, `tm`, `glmnet`, `randomForest`, `caret`, `ggplot2`, `SnowballC`, `Matrix`, `reticulate`, `keras` and `tensorflow`. The neural-network section expects a configured Python virtual environment named `r-keras`; R/Python TensorFlow compatibility must be established separately.

Place authorised copies of `worldbuilding.csv`, `scifi.csv`, `english_language.csv` and `english_learners.csv` beside `analysis.R`, then run:

```sh
Rscript analysis.R
```

Input columns include `Category`, `Title`, `Text`, `Score`, `ViewCount` and `AnswerCount`. The script writes its figures and summary tables into the working directory. `set.seed(42)` controls R random operations; this does not guarantee identical neural-network results across runtimes.

## Data and reproducibility

The course-provided datasets are omitted because their redistribution rights and original question-level attribution have not been established. There is no fabricated replacement dataset or performance claim. Full training has not been rerun for this publication. Package versions are not locked, so this is an analysis source snapshot rather than a turnkey benchmark.

## Methodological limitations

The original script constructs TF-IDF weights separately for training and test corpora, although the test vocabulary is aligned. A stronger evaluation would learn the IDF transformation on training data and apply those same weights to held-out documents. Treat existing metrics as coursework results pending that correction and a fresh evaluation. Also assess duplicate questions, class balance outside this sample and sensitivity to random splits before making generalisation claims.

## Portfolio focus

The project demonstrates text preparation, model comparison and error analysis in R. The source is preserved with a personal student identifier removed; it is not presented as production NLP software.
