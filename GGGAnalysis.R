library(tidyverse)
library(tidymodels)
library(vroom)
library(stacks)
library(patchwork)
library(DataExplorer)
library(dplyr)
library(glmnet)
library(rpart)
library(ranger)
library(bonsai)
library(lightgbm)
library(agua)
library(ggplot2)
library(embed)
library(recipes)
library(discrim)
library(themis)

# setwd("C:\\Users\\madel\\OneDrive\\Documents\\Stat 348\\GGG")


##################################################################

# read in the data
ggg_train <- vroom("./train.csv")
ggg_test <- vroom("./test.csv")

ggg_train <- ggg_train %>%
  mutate(type = as.factor(type))

full <- bind_rows(ggg_train, ggg_test)

##########################################################
# glmnet
library(dplyr)

library(ggplot2)

library(caret)
library(glmnet)
library(ranger)
library(e1071)
install.packages("clValid")
library(clValid)

train <- read.csv(file = "train.csv", header = TRUE, stringsAsFactors = FALSE)
train$Dataset <- "train"

test <- read.csv(file = "test.csv", header = TRUE, stringsAsFactors = FALSE)
test$Dataset <- "test"

full <- bind_rows(train, test)


factor_variables <- c('id', 'color', 'type', 'Dataset')
full[factor_variables] <- lapply(full[factor_variables], function(x) as.factor(x))

full <- full %>%
  mutate(hair_soul = hair_length * has_soul)

full <- full %>%
  mutate(bone_flesh = bone_length * rotting_flesh,
         bone_hair = bone_length * hair_length,
         bone_soul = bone_length * has_soul,
         flesh_hair = rotting_flesh * hair_length,
         flesh_soul = rotting_flesh * has_soul)

set.seed(100)

# Extract creature labels and remove column from dataset
creature_labels <- full$type
full2 <- full
full2$type <- NULL

# Remove categorical variables (id, color, and dataset) from dataset
full2$id <- NULL
full2$color <- NULL
full2$Dataset <- NULL

# Perform k-means clustering with 3 clusters, repeat 30 times
creature_km_1 <- kmeans(full2, 3, nstart = 30)

dunn_ckm_1 <- dunn(clusters = creature_km_1$cluster, Data = full2)

# Print results
dunn_ckm_1

train_complete <- full[full$Dataset == 'train', ]
test_complete <- full[full$Dataset == 'test', ]

myControl <- trainControl(
  method = "cv", 
  number = 10,
  repeats = 20, 
  verboseIter = TRUE
)

set.seed(10)

glm_model_2 <- train(
  type ~ bone_length + rotting_flesh + has_soul + hair_soul + bone_flesh + bone_hair + 
    bone_soul + flesh_hair + flesh_soul, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0:1,
                         lambda = seq(0.0001, 1, length = 20)),
  data = train_complete,
  trControl = myControl
)

test_complete <- test_complete %>%
  arrange(id)

# Make predicted survival values
my_prediction <- predict(glm_model_2, test_complete)

my_solution_GGG_03 <- data.frame(id = test_complete$id, Type = my_prediction)

# Write the solution to a csv file 
write.csv(my_solution_GGG_03, file = "ggg_glm.csv", row.names = FALSE)



##########################################################
# stacking
ghost_recipe <- recipe(type ~ ., data = ggg_train) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

folds <- vfold_cv(ggg_train, v = 10, strata = type)

knn_model <- nearest_neighbor(
  neighbors = tune(),
  dist_power = tune(),
  weight_func = tune()
) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(ghost_recipe)

knn_grid <- grid_space_filling(
  neighbors(range = c(1, 40)),
  dist_power(range = c(1, 3)),
  weight_func(),
  size = 25
)

knn_res <- tune_grid(
  knn_wf,
  resamples = folds,
  grid = knn_grid,
  metrics = metric_set(accuracy, mn_log_loss),
  control = control_grid(
    save_pred = TRUE,
    save_workflow = TRUE,
    control_stack = TRUE
  )
)


rf_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

rf_wf <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(ghost_recipe)

rf_grid <- grid_space_filling(
  mtry(range = c(5, 40)),
  min_n(range = c(2, 20)),
  size = 25
)

rf_res <- tune_grid(
  rf_wf,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(accuracy, mn_log_loss),
  control = control_grid(
    save_pred = TRUE,
    save_workflow = TRUE,
    control_stack = TRUE
  )
)


boost_model <- boost_tree(
  trees = 1000,
  learn_rate = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune()
) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

boost_wf <- workflow() %>%
  add_model(boost_model) %>%
  add_recipe(ghost_recipe)

boost_grid <- grid_space_filling(
  learn_rate(),
  tree_depth(range = c(2, 8)),
  min_n(range = c(2, 20)),
  loss_reduction(),
  size = 25
)

boost_res <- tune_grid(
  boost_wf,
  resamples = folds,
  grid = boost_grid,
  metrics = metric_set(accuracy, mn_log_loss),
  control = control_grid(
    save_pred = TRUE,
    save_workflow = TRUE,
    control_stack = TRUE
  )
)


ghost_stack <- 
  stacks() %>%
  add_candidates(knn_res) %>%
  add_candidates(rf_res) %>%
  add_candidates(boost_res) %>%
  blend_predictions(metric = mn_log_loss) %>%
  fit_members()

final_stack_fit <- 
  ghost_stack %>%
  predict(new_data = ggg_test, type = "class")

prob_preds <- predict(ghost_stack, new_data = ggg_test, type = "prob")

kaggle_submission <- tibble(
  id = ggg_test$id,
  type = final_stack_fit$.pred_class
)

vroom_write(kaggle_submission, "./ggg_stack.csv")

# get rid of IDs
# color as a factor
# linear svm
# now 0.735
# regression trees

# step normalize
# step_poly




######################################################################
#### KNN -- 0.35538

knn_model <- nearest_neighbor(
  neighbors = tune(),
  dist_power = tune(),
  weight_func = tune()
) %>%
  set_mode("classification") %>%
  set_engine("kknn")


tuning_grid <- grid_regular(
  neighbors(range = c(1, 30)),
  dist_power(range = c(1, 3)),
  levels = 7
)

metrics = metric_set(accuracy, mn_log_loss, kap)


my_recipe <- recipe(type ~ ., data = ggg_train) %>%
  step_other(all_nominal_predictors(), threshold = 0.1) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_predictors(), num_comp = tune())


prep <- prep(my_recipe)
baked <- bake(prep, new_data = ggg_train)


knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)


folds <- vfold_cv(ggg_train, v = 10, repeats = 3)

CV_results <- knn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=ggg_train)

ggg_predictions <- final_wf %>%
  predict(new_data = ggg_test, type='class')

kaggle_submission <- tibble(
  id = ggg_test$id,
  type = ggg_predictions$.pred_class)


vroom_write(x = kaggle_submission, file = "./ggg_knn.csv", delim = ",")

##########################################################################
# new KNN
knn_model <- nearest_neighbor(
  neighbors = tune(), 
  dist_power = tune(),
  weight_func = tune()
) %>%
  set_mode("classification") %>%
  set_engine("kknn")

my_recipe <- recipe(type ~ ., data = ggg_train) %>%
  step_other(all_nominal_predictors(), threshold = 0.1) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

tuning_grid <- grid_space_filling(
  neighbors(range = c(1, 40)),
  dist_power(range = c(1, 3)),
  weight_func(),
  size = 30
)


folds <- vfold_cv(ggg_train, v = 10, repeats = 2)

CV_results <- tune_grid(
  knn_wf,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(accuracy, mn_log_loss, kap)
)

bestTune <- CV_results %>%
  select_best(metric = "accuracy")

final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=ggg_train)

ggg_predictions <- final_wf %>%
  predict(new_data = ggg_test, type='class')

kaggle_submission <- tibble(
  id = ggg_test$id,
  type = ggg_predictions$.pred_class)


vroom_write(x = kaggle_submission, file = "./ggg_knn1.csv", delim = ",")


############################################################

### LOGISTIC REGRESSION - bad

my_recipe <- recipe(type ~ ., data = ggg_train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_nominal_predictors())

prep <- prep(my_recipe)
baked <- bake(prep, new_data = ggg_train)

logRegModel <- logistic_reg() %>%
  set_engine("glm")

logReg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data = ggg_train)

ggg_predictions <- predict(logReg_wf,
                              new_data=ggg_test,
                              type="class")

kaggle_submission <- tibble(
  id = ggg_test$id,
  type = ggg_predictions$.pred_class)

vroom_write(x = kaggle_submission, file = "./ggg_logistic.csv", delim = ",")

##################################################################

# STACKING
library(h2o)
h2o.init()

# Convert data frames to H2O frames
train_h2o <- as.h2o(ggg_train)
test_h2o  <- as.h2o(ggg_test)

# Define target and predictors
y <- "type"      # e.g. ghost/ghoul/goblin
x <- setdiff(names(ggg_train), y)

# Run AutoML (includes Naive Bayes, trees, etc.)
auto_model <- h2o.automl(
  x = x,
  y = y,
  training_frame = train_h2o,
  max_models = 20,
  max_runtime_secs = 600,
  seed = 123,
  balance_classes = TRUE,
  nfolds = 5
)

# Get leaderboard and best stacked model
lb <- auto_model@leaderboard
print(lb)

best_model <- auto_model@leader

# Predict on test set
ggg_preds <- h2o.predict(best_model, test_h2o)

kaggle_submission <- tibble(
  id = ggg_test$id,
  type = ggg_preds$.pred_class)

vroom_write(x = kaggle_submission, file = "./ggg_stack_naive_trees.csv", delim = ",")

####################################################

# MIX GLM AND REGRESSION TREES
install.packages("caret")
library(caret)
library(glmnet)
library(ranger)
library(e1071)
library(clValid)

full <- full %>%
  mutate(hair_soul = hair_length * has_soul)

myControl <- trainControl(
  method = "cv", 
  number = 10,
  repeats = 20, 
  verboseIter = TRUE
)

set.seed(10)

glm_model_2 <- train(
  type ~ bone_length + rotting_flesh + hair_length + has_soul + hair_soul + bone_flesh + bone_hair + 
    bone_soul + flesh_hair + flesh_soul, 
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0:1,
                         lambda = seq(0.0001, 1, length = 20)),
  data = ggg_train,
  trControl = myControl
)


test_complete <- test_complete %>%
  arrange(id)

# Make predicted survival values
my_prediction <- predict(glm_model_2, test_complete)


my_solution_GGG_03 <- data.frame(id = test_complete$id, Type = my_prediction)

# Write the solution to a csv file 
write.csv(my_solution_GGG_03, file = "my_solution_GGG_03.csv", row.names = FALSE)




