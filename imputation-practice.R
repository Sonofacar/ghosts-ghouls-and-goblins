# Libraries
library(tidyverse)
library(tidymodels)
library(vroom)

# Read in data
na_data <- vroom("trainWithMissingValues.csv") %>%
  mutate(color = factor(color))
full_data <- vroom("train.csv") %>%
  mutate(color = factor(color))

# See the distribution of NA values
for (col in colnames(na_data)) {
  num <- na_data[[col]] %>%
    is.na() %>%
    sum()
  paste(col, ": ", num, sep = "") %>%
    write(file = stdout())
}

# Check if any rows have multiple missing values
for (row in length(na_data)) {
  nas <- na_data[row, ] %>%
    is.na() %>%
    sum()
  if (nas > 1) {
    print(nas, file = stdout())
  }
}

# Create a recipe
recipe <- recipe(type ~ ., data = na_data) %>%
  step_impute_knn(bone_length, impute_with = imp_vars(id,
                                                      has_soul,
                                                      color),
                  neighbors = 20) %>%
  step_impute_knn(rotting_flesh, impute_with = imp_vars(id,
                                                        bone_length,
                                                        has_soul,
                                                        color),
                  neighbors = 20) %>%
  step_impute_knn(hair_length, impute_with = imp_vars(id,
                                                      rotting_flesh,
                                                      bone_length,
                                                      has_soul,
                                                      color),
                  neighbors = 20)
prepped_recipe <- prep(recipe)
imputed_data <- bake(prepped_recipe, new_data = na_data)

# Compute RMSE
rmse_vec(full_data[is.na(na_data)], imputed_data[is.na(na_data)])

