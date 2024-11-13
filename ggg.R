# Libraries
library(tidyverse)
library(tidymodels)
library(discrim)
library(vroom)
library(doParallel)
library(themis)
library(embed)
library(tensorflow)
library(reticulate)
library(keras3)

# Set up parallelization
num_cores <- 4
cl <- makePSOCKcluster(num_cores)

# Read the data
train_dirty <- vroom("train.csv") %>%
  mutate(color = factor(color)) %>%
  mutate(type = factor(type))
test_dirty <- vroom("test.csv")

# Recipe
recipe <- recipe(type ~ ., data = train_dirty) %>%
  update_role(id, new_role = "id") %>%
  #step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

# Create folds in the data
folds <- vfold_cv(train_dirty, v = 10, repeats = 1)

###############
# Naive Bayes #
###############

# Create a model
bayes_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

# Create the workflow
bayes_workflow <- workflow() %>%
  add_model(bayes_model) %>%
  add_recipe(recipe)

# Set up parallelization
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# Tuning
bayes_tuning_grid <- grid_regular(Laplace(), smoothness(), levels = 10)
bayes_cv_results <- bayes_workflow %>%
  tune_grid(resamples = folds,
            grid = bayes_tuning_grid,
            metrics = metric_set(accuracy, roc_auc))
stopCluster(cl)

# Get the best tuning parameters
bayes_besttune <- bayes_cv_results %>%
  select_best(metric = "accuracy")

# Fit and make predictions
bayes_fit <- bayes_workflow %>%
  finalize_workflow(bayes_besttune) %>%
  fit(data = train_dirty)
bayes_predictions <- predict(bayes_fit,
                             new_data = test_dirty,
                             type = "class")$.pred_class

# Write output
bayes_output <- tibble(id = test_dirty$id,
                       type = bayes_predictions)
vroom_write(bayes_output, "naive_bayes.csv", delim = ",")

##################
# Neural Network #
##################

# Create a model
nn_model <- mlp(hidden_units = tune(), epochs = 100) %>%
  set_mode("classification") %>%
  set_engine("keras")

# Create the workflow
nn_workflow <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(recipe)

# Tuning
nn_tuning_grid <- grid_regular(hidden_units(), levels = 20)
nn_cv_results <- nn_workflow %>%
  tune_grid(resamples = folds,
            grid = nn_tuning_grid,
            metrics = metric_set(accuracy, roc_auc))

# Get the best tuning parameters
nn_besttune <- nn_cv_results %>%
  select_best(metric = "accuracy")

# Fit and make predictions
nn_fit <- nn_workflow %>%
  finalize_workflow(nn_besttune) %>%
  fit(data = train_dirty)
nn_predictions <- predict(nn_fit,
                          new_data = test_dirty,
                          type = "class")$.pred_class

# Write output
nn_output <- tibble(id = test_dirty$id,
                    type = nn_predictions)
vroom_write(nn_output, "neural_net.csv", delim = ",")

