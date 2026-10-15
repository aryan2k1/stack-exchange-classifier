# =============================================================================
# RESM3002 Advanced Social Data Science — Assignment 2026
# Stack Exchange question category classifier
# =============================================================================
# dataset summary (confirmed from the files):
#   worldbuilding.csv   -> world_building  (2,000 rows, avg text 1,615 chars)
#   scifi.csv           -> scifi           (2,000 rows, avg text 1,072 chars)
#   english_language.csv-> english_language(2,000 rows, avg text   975 chars)
#   english_learners.csv-> english_learners(2,000 rows, avg text   800 chars)
#   total: 8000 rows, 4 perfectly balanced classes, no missing values
# =============================================================================


# =============================================================================
# 0. setup & libraries
# =============================================================================

set.seed(42) # Set once globally - also re-set before specific random operations

# install once if you don't have these, then comment back out:
# install.packages(c("tidyverse", "tm", "glmnet", "randomForest",
#                    "caret", "ggplot2", "SnowballC", "Matrix"))

library(tidyverse)
library(tm)
library(glmnet)
library(randomForest)
library(caret)
library(ggplot2)
library(SnowballC)
library(Matrix)


# =============================================================================
# 1. data loading
# =============================================================================

# point R at the folder with the 4 CSVs
# setwd("~/path/to/your/data")    # Mac/Linux

# pull in all 4 CSVs — filenames match what's on Blackboard
world_building    <- read.csv("worldbuilding.csv",        stringsAsFactors = FALSE)
scifi             <- read.csv("scifi.csv",             stringsAsFactors = FALSE)
english_language  <- read.csv("english_language.csv",  stringsAsFactors = FALSE)
english_learners  <- read.csv("english_learners.csv",  stringsAsFactors = FALSE)

# stack everything into one dataframe — 8000 rows total
data <- bind_rows(world_building, scifi, english_language, english_learners)

# Category as a factor — it's the outcome we're predicting
data$Category <- as.factor(data$Category)

# sanity check — should be 2000 per category
cat("=== Data loaded ===\n")
cat("Total rows:", nrow(data), "\n")
print(table(data$Category))

# join Title + Text — the title often carries domain-specific words
data$FullText <- paste(data$Title, data$Text, sep = " ")


# =============================================================================
# 2. train / test split (stratified 80/20)
# =============================================================================
# stratified — keeps the 25/25/25/25 class balance in both sets

set.seed(42)
train_idx  <- createDataPartition(data$Category, p = 0.8, list = FALSE)
train_data <- data[train_idx, ]
test_data  <- data[-train_idx, ]

cat("\n=== Train/test split ===\n")
cat("Training rows:", nrow(train_data), "\n")
cat("Test rows:    ", nrow(test_data), "\n")
cat("Train category distribution:\n")
print(prop.table(table(train_data$Category)))


# =============================================================================
# 3. descriptive stats 
# =============================================================================

# --- 3a. class balance table ---
class_table <- data %>%
  count(Category) %>%
  mutate(Percentage = round(n / sum(n) * 100, 1))
print(class_table)
# perfectly balanced — no class-weighting fix needed,

# --- 3b. summary stats for the numeric vars ---
summary_stats <- data %>%
  group_by(Category) %>%
  summarise(
    n            = n(),
    mean_score   = round(mean(Score,       na.rm = TRUE), 2),
    median_score = round(median(Score,     na.rm = TRUE), 2),
    mean_views   = round(mean(ViewCount,   na.rm = TRUE), 0),
    mean_answers = round(mean(AnswerCount, na.rm = TRUE), 2),
    avg_chars    = round(mean(nchar(FullText), na.rm = TRUE), 0),
    sd_chars     = round(sd(nchar(FullText),   na.rm = TRUE), 0),
    .groups = "drop"
  )
print(summary_stats)


# --- 3c. boxplots of numeric vars by category ---
data$text_length <- nchar(data$FullText)

data_long <- data %>%
  select(Category, Score, ViewCount, AnswerCount, text_length) %>%
  pivot_longer(cols = -Category, names_to = "Variable", values_to = "Value")

ggplot(data_long, aes(x = Category, y = Value, fill = Category)) +
  geom_boxplot(outlier.alpha = 0.2, outlier.size = 0.5) +
  facet_wrap(~Variable, scales = "free_y") +
  labs(title = "Distribution of Variables by Category",
       x = NULL, y = NULL) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 30, hjust = 1, size = 8))
ggsave("fig_01_boxplots_by_category.png", width = 9, height = 6, dpi = 150)

# --- 3d. score distribution — the headline pattern ---
ggplot(data, aes(x = Score, fill = Category)) +
  geom_histogram(bins = 40, alpha = 0.7) +
  facet_wrap(~Category, scales = "free_y") +
  labs(title = "Score Distribution by Category",
       x = "Question Score (upvotes)", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave("fig_02_score_distribution.png", width = 9, height = 5, dpi = 150)

# --- 3e. text length distribution ---
ggplot(data, aes(x = text_length, fill = Category)) +
  geom_density(alpha = 0.5) +
  labs(title = "Question Text Length by Category",
       x = "Character count (Title + Text combined)", y = "Density") +
  theme_minimal()
ggsave("fig_03_text_length_distribution.png", width = 8, height = 4, dpi = 150)


# =============================================================================
# 4. text preprocessing  (same pipeline shared across all three models)
# =============================================================================

# standard text cleaning — lowercase, drop punctuation/numbers/stopwords,
# then Porter stem ("running" → "run", "worlds" → "world")
build_corpus <- function(text_vector) {
  corpus <- VCorpus(VectorSource(text_vector))
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopwords("english"))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, stemDocument)
  return(corpus)
}

cat("\n=== Building text features ===\n")
cat("Building training corpus (this takes ~1 minute)...\n")
train_corpus <- build_corpus(train_data$FullText)

# build the TF-IDF document-term matrix
# sparse=0.995 keeps terms in at least 0.5% of docs
# cuts rare-word noise and speeds the models up a lot
dtm_train <- DocumentTermMatrix(
  train_corpus,
  control = list(weighting = weightTfIdf)
)
dtm_train <- removeSparseTerms(dtm_train, sparse = 0.995)

cat("Vocabulary size:", ncol(dtm_train), "terms\n")

# apply the SAME vocab to the test set — this is the leakage fix
# dictionary = Terms(dtm_train) is what actually enforces it
cat("Building test corpus...\n")
test_corpus <- build_corpus(test_data$FullText)
dtm_test <- DocumentTermMatrix(
  test_corpus,
  control = list(
    weighting  = weightTfIdf,
    dictionary = Terms(dtm_train)   # <-- critical: no data leakage
  )
)

# turn into plain matrices
X_train_text <- as.matrix(dtm_train)
X_test_text  <- as.matrix(dtm_test)

# --- numeric features: scale using train means and SDs only — same leakage logic ---
numeric_vars <- c("Score", "ViewCount", "AnswerCount", "text_length")

train_data$text_length <- nchar(train_data$FullText)
test_data$text_length  <- nchar(test_data$FullText)

train_numeric_raw <- train_data %>%
  select(all_of(numeric_vars)) %>%
  mutate(across(everything(), ~replace_na(., 0)))

scaled_obj   <- scale(train_numeric_raw)
numeric_train <- as.matrix(scaled_obj)
train_means  <- attr(scaled_obj, "scaled:center")
train_sds    <- attr(scaled_obj, "scaled:scale")

test_numeric_raw <- test_data %>%
  select(all_of(numeric_vars)) %>%
  mutate(across(everything(), ~replace_na(., 0)))

numeric_test <- sweep(as.matrix(test_numeric_raw), 2, train_means, "-")
numeric_test <- sweep(numeric_test,                2, train_sds,   "/")

# join TF-IDF + numeric into one feature matrix per set
X_train <- cbind(X_train_text, numeric_train)
X_test  <- cbind(X_test_text,  numeric_test)
y_train <- train_data$Category
y_test  <- test_data$Category

cat("Final feature matrix dimensions:", dim(X_train), "\n")
cat("Numeric features included:", numeric_vars, "\n")


# =============================================================================
# 5. top words per category 
# =============================================================================

get_top_words <- function(dtm_matrix, labels, category, n = 15) {
  idx      <- which(labels == category)
  col_sums <- colSums(dtm_matrix[idx, ])
  top      <- sort(col_sums, decreasing = TRUE)[1:n]
  data.frame(term = names(top), tfidf_sum = round(top, 3), category = category)
}

top_words_all <- bind_rows(lapply(levels(y_train), function(cat) {
  get_top_words(X_train_text, y_train, cat)
}))

# print the top words — useful to drop into the report
for (category_name in levels(y_train)) {
  cat("\n--- Top words:", category_name, "---\n")
  print(top_words_all %>% filter(category == category_name) %>% select(term, tfidf_sum))
}

# plot top words per category
top_words_all %>%
  group_by(category) %>%
  top_n(12, tfidf_sum) %>%
  ggplot(aes(x = reorder(term, tfidf_sum), y = tfidf_sum, fill = category)) +
  geom_bar(stat = "identity") +
  facet_wrap(~category, scales = "free") +
  coord_flip() +
  labs(title = "Top 12 TF-IDF Terms per Category",
       x = "Term", y = "Total TF-IDF weight") +
  theme_minimal() +
  theme(legend.position = "none", axis.text.y = element_text(size = 8))
ggsave("fig_04_top_tfidf_terms.png", width = 10, height = 7, dpi = 150)


# =============================================================================
# 6a. LASSO — multinomial classifier  
# =============================================================================

cat("\n=== Model 1: LASSO ===\n")

# fit a single LASSO first so we can plot the coefficient paths
lasso_single <- glmnet(
  x = X_train,
  y = y_train,
  family          = "multinomial",
  type.multinomial = "group",  # same word set used across all categories
  alpha           = 1          # alpha=1 is LASSO; alpha=0 is Ridge
)
png("fig_05_lasso_coef_paths.png", width = 800, height = 500)
plot(lasso_single, xvar = "lambda", label = FALSE,
     main = "LASSO: Coefficient paths as lambda varies")
dev.off()

# 5-fold CV to pick lambda — shrinks useless term coefficients to exactly zero
# nfolds=5 — decent tradeoff between speed and stability
cat("Running 5-fold cross-validation (may take 3-5 minutes)...\n")
set.seed(42)
cv_lasso <- cv.glmnet(
  x = X_train,
  y = y_train,
  family          = "multinomial",
  type.multinomial = "group",
  nfolds          = 5,
  alpha           = 1,
  trace.it        = TRUE
)

png("fig_06_lasso_cv_lambda.png", width = 800, height = 500)
plot(cv_lasso, main = "LASSO: Cross-validation error by lambda")
dev.off()
cat("Best lambda (min):  ", cv_lasso$lambda.min, "\n")
cat("Best lambda (1se):  ", cv_lasso$lambda.1se, "\n")
# lambda.1se picks the simplest model within 1 SE of the best — less likely to overfit

# predict on test
lasso_pred <- predict(cv_lasso, newx = X_test, s = "lambda.1se", type = "class")
lasso_pred <- factor(lasso_pred[, 1], levels = levels(y_test))

# evaluate
lasso_cm <- confusionMatrix(lasso_pred, y_test)
cat("\n--- LASSO confusion matrix ---\n")
print(lasso_cm)

# how many words did LASSO actually keep? (interpretability)
lasso_coefs  <- coef(cv_lasso, s = "lambda.1se")
n_nonzero <- length(unique(unlist(lapply(lasso_coefs, function(m) {
  rn <- rownames(m)
  rn[as.vector(m) != 0 & rn != "(Intercept)"]
}))))
cat("\nUnique non-zero features selected across all classes:", n_nonzero, "\n")
# this is LASSO's automatic feature selection in action

# pull and plot the strongest predictive terms per category
extract_top_terms <- function(coef_list, n = 15) {
  bind_rows(lapply(names(coef_list), function(cat_name) {
    coefs_df <- as.data.frame(as.matrix(coef_list[[cat_name]]))
    colnames(coefs_df) <- "coef"
    coefs_df$term <- rownames(coefs_df)
    coefs_df %>%
      filter(term != "(Intercept)", coef != 0) %>%
      arrange(desc(abs(coef))) %>%
      head(n) %>%
      mutate(category = cat_name)
  }))
}

top_lasso_terms <- extract_top_terms(lasso_coefs)
write.csv(top_lasso_terms, "lasso_top_terms.csv", row.names = FALSE)

ggplot(top_lasso_terms, aes(x = reorder(term, coef), y = coef, fill = coef > 0)) +
  geom_bar(stat = "identity") +
  facet_wrap(~category, scales = "free") +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#1D9E75", "FALSE" = "#D85A30"),
                    labels = c("TRUE" = "Positive (predicts this category)",
                               "FALSE" = "Negative (predicts against)")) +
  labs(title = "Top LASSO Coefficients by Category",
       subtitle = "Positive = word predicts this category; Negative = word predicts against it",
       x = NULL, y = "Coefficient", fill = NULL) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 8))
ggsave("fig_07_lasso_top_terms.png", width = 11, height = 7, dpi = 150)


# =============================================================================
# 6b. Random Forest 
# =============================================================================

cat("\n=== Model 2: Random Forest ===\n")
cat("Fitting 500 trees (may take 5-10 minutes)...\n")

set.seed(42)
rf_model <- randomForest(
  x           = X_train,
  y           = y_train,
  ntree       = 500,
  importance  = TRUE,   # needed for variable importance plot
  do.trace    = 100     # print OOB error every 100 trees
)

print(rf_model)
# OOB (Out-Of-Bag) error is an unbiased estimate of test error

# plot OOB error vs ntree
png("fig_08_rf_oob_error.png", width = 800, height = 500)
plot(rf_model,
     main = "Random Forest: OOB error rate by number of trees",
     lwd  = 1.5)
legend("topright", colnames(rf_model$err.rate),
       col = 1:5, lty = 1:3, cex = 0.7)
dev.off()
# for the report: lines flattening before 500 trees validates ntree=500

# predict on test
rf_pred <- predict(rf_model, newdata = X_test)
rf_cm   <- confusionMatrix(rf_pred, y_test)
cat("\n--- Random Forest confusion matrix ---\n")
print(rf_cm)

# variable importance — which words actually matter most
png("fig_09_rf_variable_importance.png", width = 800, height = 600)
varImpPlot(rf_model, n.var = 20, type = 1,
           main = "Random Forest: Top 20 predictive variables")
dev.off()

# save importance scores
importance_df <- as.data.frame(importance(rf_model, type = 1)) %>%
  rownames_to_column("term") %>%
  arrange(desc(MeanDecreaseAccuracy)) %>%
  head(20)
cat("\nTop 20 important features (Random Forest):\n")
print(importance_df)


# =============================================================================
# 6c. Neural Network (keras3) 
# =============================================================================
# Run: install.packages("keras"); keras::install_keras()

reticulate::use_virtualenv("r-keras", required = TRUE)
library(keras)
library(tensorflow)  # needed for tf$cast()
tf <- tensorflow::tf

# NNs want integer token sequences, not TF-IDF — embedding layer needs IDs
# the limits below keep training time sensible
MAX_FEATURES  <- 10000L  # top N words by frequency
MAX_LEN       <- 200L    # truncate/pad all texts to this many tokens
EMBEDDING_DIM <- 64L     # size of word embedding vectors
BATCH_SIZE    <- 32L
EPOCHS        <- 10L

# encode categories as 0-indexed ints — keras wants this
label_levels <- levels(y_train)
y_train_int  <- as.integer(y_train) - 1L
y_test_int   <- as.integer(y_test)  - 1L

# text vectoriser — fit on training text only, same leakage logic as TF-IDF
text_vectorizer <- layer_text_vectorization(
  max_tokens             = MAX_FEATURES,
  output_sequence_length = MAX_LEN,
  output_mode            = "int"
)
text_vectorizer$adapt(train_data$FullText)  # fit on training data only


X_train_nn <- tf$cast(text_vectorizer(train_data$FullText), dtype = tf$int32)
X_test_nn  <- tf$cast(text_vectorizer(test_data$FullText),  dtype = tf$int32)
y_train_tf <- tf$constant(y_train_int, dtype = tf$int32)

nn_model <- keras_model_sequential(name = "text_classifier")
nn_model$add(layer_embedding(
  input_dim  = MAX_FEATURES + 1L,
  output_dim = EMBEDDING_DIM,
  name       = "embedding"                 # input_length removed (deprecated in keras3)
))
nn_model$add(layer_global_average_pooling_1d(name = "pooling"))   # mean over sequence
nn_model$add(layer_dense(units = 64L, activation = "relu",    name = "dense1"))
nn_model$add(layer_dropout(rate = 0.3,                         name = "dropout1"))  # regularisation
nn_model$add(layer_dense(units = 32L, activation = "relu",    name = "dense2"))
nn_model$add(layer_dropout(rate = 0.2,                         name = "dropout2"))
nn_model$add(layer_dense(units = 4L,  activation = "softmax", name = "output"))    # 4 classes

nn_model$summary()

nn_model$compile(
  optimizer = keras$optimizers$Adam(learning_rate = 0.001),
  loss      = "sparse_categorical_crossentropy",
  metrics   = list("accuracy")
)

# train — validation_split off
set.seed(42)
nn_history <- nn_model$fit(
  x          = X_train_nn,
  y          = y_train_tf,
  epochs     = EPOCHS,
  batch_size = BATCH_SIZE,
  verbose    = 1L
)

# plot accuracy and loss across epochs
nn_hist_df <- data.frame(
  epoch    = seq_along(nn_history$history$accuracy),
  accuracy = nn_history$history$accuracy,
  loss     = nn_history$history$loss
)
ggplot(nn_hist_df, aes(x = epoch)) +
  geom_line(aes(y = accuracy, colour = "accuracy"), linewidth = 1) +
  geom_line(aes(y = loss,     colour = "loss"),     linewidth = 1) +
  scale_colour_manual(values = c("accuracy" = "#1D9E75", "loss" = "#D85A30")) +
  labs(title = "Neural Network Training History",
       x = "Epoch", y = "Value", colour = "Metric") +
  theme_minimal()
ggsave("fig_10_nn_training_history.png", width = 8, height = 5, dpi = 150)
# for the report: loss falling and accuracy rising = the network is learning

# evaluate on the held-out test set
nn_eval <- nn_model$evaluate(
  X_test_nn,
  tf$constant(y_test_int, dtype = tf$int32),
  verbose = 0L
)
cat("\nNeural Network test accuracy:", round(nn_eval[[2]], 4), "\n")

# get predicted classes
nn_probs    <- nn_model$predict(X_test_nn, verbose = 0L)
nn_pred_int <- apply(nn_probs, 1, which.max) - 1L
nn_pred     <- factor(label_levels[nn_pred_int + 1L], levels = levels(y_test))

nn_cm <- confusionMatrix(nn_pred, y_test)
cat("\n--- Neural Network confusion matrix ---\n")
print(nn_cm)


# =============================================================================
# 7. model comparison table
# =============================================================================

extract_metrics <- function(cm, model_name) {
  data.frame(
    Model            = model_name,
    Accuracy         = round(cm$overall["Accuracy"], 4),
    Macro_Precision  = round(mean(cm$byClass[, "Precision"], na.rm = TRUE), 4),
    Macro_Recall     = round(mean(cm$byClass[, "Recall"],    na.rm = TRUE), 4),
    Macro_F1         = round(mean(cm$byClass[, "F1"],        na.rm = TRUE), 4)
  )
}

comparison <- bind_rows(
  extract_metrics(lasso_cm, "LASSO"),
  extract_metrics(rf_cm,    "Random Forest"),
  extract_metrics(nn_cm,    "Neural Network")
)
cat("\n=== MODEL COMPARISON ===\n")
print(comparison)
write.csv(comparison, "model_comparison_table.csv", row.names = FALSE)
cat("Model comparison saved to model_comparison_table.csv\n")

ggplot(comparison %>% pivot_longer(-Model, names_to = "Metric", values_to = "Value"),
       aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity") +
  facet_wrap(~Metric) +
  ylim(0, 1) +
  labs(title = "Model comparison across metrics", x = NULL, y = "Score") +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 20, hjust = 1))
ggsave("fig_11_model_comparison.png", width = 9, height = 5, dpi = 150)


# =============================================================================
# 8. confusion matrix heatmaps
# =============================================================================

plot_cm_heatmap <- function(cm, model_name) {
  cm_df <- as.data.frame(cm$table) %>%
    setNames(c("Predicted", "Reference", "Freq")) %>%
    group_by(Reference) %>%
    mutate(Prop = Freq / sum(Freq)) %>%
    ungroup()

  ggplot(cm_df, aes(x = Reference, y = Predicted, fill = Prop)) +
    geom_tile(colour = "white", linewidth = 0.5) +
    geom_text(aes(label = paste0(Freq, "\n(", round(Prop * 100, 1), "%)")),
              size = 3) +
    scale_fill_gradient(low = "#E6F1FB", high = "#185FA5",
                        limits = c(0, 1), name = "Proportion") +
    labs(title = paste("Confusion Matrix:", model_name),
         x = "True category", y = "Predicted category") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
}

plot_cm_heatmap(lasso_cm, "LASSO")
ggsave("fig_12_cm_lasso.png", width = 7, height = 5, dpi = 150)

plot_cm_heatmap(rf_cm, "Random Forest")
ggsave("fig_13_cm_random_forest.png", width = 7, height = 5, dpi = 150)

plot_cm_heatmap(nn_cm, "Neural Network")
ggsave("fig_14_cm_neural_network.png", width = 7, height = 5, dpi = 150)


# =============================================================================
# 9. per-class performance
# =============================================================================

# compare per-class F1 across the three models
per_class_f1 <- function(cm, model_name) {
  f1_df <- as.data.frame(cm$byClass) %>%
    rownames_to_column("Class") %>%
    mutate(
      Class = str_remove(Class, "Class: "),
      Model = model_name
    ) %>%
    select(Class, Model, F1)
  f1_df
}

f1_all <- bind_rows(
  per_class_f1(lasso_cm, "LASSO"),
  per_class_f1(rf_cm,    "Random Forest"),
  per_class_f1(nn_cm,    "Neural Network")
)
write.csv(f1_all, "per_class_f1_table.csv", row.names = FALSE)
cat("Per-class F1 saved to per_class_f1_table.csv\n")

ggplot(f1_all, aes(x = Class, y = F1, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Per-category F1 score by model",
       x = NULL, y = "F1 score") +
  ylim(0, 1) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
ggsave("fig_15_per_class_f1.png", width = 8, height = 5, dpi = 150)

# =============================================================================
# END OF SCRIPT
# =============================================================================
