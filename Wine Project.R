if (!require(dplyr)) install.packages('dplyr')
library(dplyr)
if (!require(ggplot2)) install.packages('ggplot2')
library(ggplot2)
if (!require(ggthemes)) install.packages('ggthemes')
library(ggthemes)
if (!require(caret)) install.packages('caret')
library(caret)
if (!require(e1071)) install.packages('e1071')
library(e1071)

options(digits = 3)

#Wine Data from UCI
winequality.white <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")
winequality.red <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

#Explore the Dataset's Structure
str(winequality.white)
str(winequality.red)

#Merge Datasets 
winequality.white <- data.frame(winequality.white, color = "white")
winequality.red <- data.frame(winequality.red, color = "red")
wine <- rbind(winequality.white, winequality.red)

#Turn Quality into a Factor
wine$quality = as.factor(wine$quality)
str(wine)

#Split into Training and Test Sets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = wine$quality, times = 1, p = 0.1, list = FALSE)
train_set <- wine[-test_index,]
test_set <- wine[test_index,]

#Explore Training Dataset
str(train_set)
avg_rating <- mean(wine$quality)
train_set %>% ggplot(aes(quality)) + geom_bar() +
  ggtitle("Quality Ratings") +
  theme_economist() +
  ggsave(filename = "./figure01.png")

#Visualization of Variables to spot if any are impactful
train_set %>% ggplot(aes(quality, fixed.acidity)) + geom_boxplot(aes(group=quality)) +
  ggtitle("Fixed Acid Levels by Quality") + theme_economist() +
  ggsave(filename = "./figure02.png")
train_set %>% ggplot(aes(quality, volatile.acidity)) + geom_boxplot(aes(group=quality)) +
  ggtitle("Volatile Acids Levels by Quality") + theme_economist()+
  ggsave(filename = "./figure03.png")
train_set %>% ggplot(aes(quality, citric.acid)) + geom_boxplot(aes(group=quality))+
  ggtitle("Citric Acid Levels by Quality") + theme_economist()+
  ggsave(filename = "./figure04.png")
train_set %>% ggplot(aes(quality, residual.sugar)) + geom_boxplot(aes(group=quality))+
  ggtitle("Sugar Levels by Quality") + theme_economist()+
  ggsave(filename = "./figure05.png")
train_set %>% ggplot(aes(quality, chlorides)) + geom_boxplot(aes(group=quality))+
  ggtitle("Chloride Levels by Quality") + theme_economist()+
  ggsave(filename = "./figure06.png")
train_set %>% ggplot(aes(quality, free.sulfur.dioxide)) + geom_boxplot(aes(group=quality))+
  ggtitle("Free SO2 Levels by Quality") + theme_economist()+
  ggsave(filename = "./figure07.png")
train_set %>% ggplot(aes(quality, total.sulfur.dioxide)) + geom_boxplot(aes(group=quality))+
  ggtitle("Total SO2 Levels by Quality") + theme_economist()+
  ggsave(filename = "./figure08.png")
train_set %>% ggplot(aes(quality, density)) + geom_boxplot(aes(group=quality))+
  ggtitle("Density Levels by Quality") + theme_economist()+
  ggsave(filename = "./figure09.png")
train_set %>% ggplot(aes(quality, pH)) + geom_boxplot(aes(group=quality))+
  ggtitle("pH Levels by Quality") + theme_economist()+
  ggsave(filename = "./figure10.png")
train_set %>% ggplot(aes(quality, sulphates)) + geom_boxplot(aes(group=quality))+
  ggtitle("Sulphate Levels by Quality") + theme_economist()+
  ggsave(filename = "./figure11.png")
train_set %>% ggplot(aes(quality, alcohol)) + geom_boxplot(aes(group=quality))+
  ggtitle("Alcohol Levels by Quality") + theme_economist()+
  ggsave(filename = "./figure12.png")

#Simply Guess 6 for everything
mean(train_set$quality == "6")

#Try KNN to see accuracy as a baseline
fit_knn <- train(quality ~ . , method = "knn" , data = train_set, trControl = trainControl(method = "cv", number = 10))
fit_knn

#RFE to see Accuracy & Variable Importance
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
rfe_train <- rfe(train_set[,1:11], train_set[,12], sizes=c(1:11), rfeControl=control)
rfe_train
predictors(rfe_train)
plot(rfe_train, type=c("g", "o"))

#Try some different models with top 5 factors
models <- c("lda", "treebag", "naive_bayes", "knn", "svmLinear", "gamLoess", "multinom", "rf", "lvq", "gbm", "svmRadial", "C5.0")
set.seed(1, sample.kind = "Rounding")
fits_cv <- lapply(models, function(model){ 
  print(model)
  train(quality ~ alcohol + volatile.acidity + free.sulfur.dioxide + sulphates + chlorides, method = model, data = train_set, trControl = trainControl(method = "cv", number = 10))
})
results <- resamples(fits_cv)
summary(results)
dotplot(results)

#Trim Models to Top 5 with top 5 models and top 5 factors
models_top_5 <- c("treebag", "rf", "gbm", "svmRadial", "C5.0")
set.seed(1, sample.kind = "Rounding")
fits_top_5 <- lapply(models_top_5, function(model){ 
  print(model)
  train(quality ~ alcohol + volatile.acidity + free.sulfur.dioxide + sulphates + chlorides, method = model, data = train_set, trControl = trainControl(method = "cv", number = 10))
})
results_top_5 <- resamples(fits_top_5)
summary(results_top_5)
dotplot(results_top_5)

#Compare top 5 factor results against all 12
models_5_12 <- c("treebag", "rf", "gbm", "svmRadial", "C5.0")
set.seed(1, sample.kind = "Rounding")
fits_5_12 <- lapply(models_5_12, function(model){ 
  print(model)
  train(quality ~ ., method = model, data = train_set, trControl = trainControl(method = "cv", number = 10))
})
results_5_12 <- resamples(fits_5_12)
summary(results_5_12)
dotplot(results_5_12)

# See if models with good results are over-correlated
cor_5_12 <- modelCor(results_5_12)
heatmap(cor_5_12)

#Build the Ensemble using Random Foest - Results in Error
set.seed(1, sample.kind = "Rounding")
ensemble_models <- caretList(quality~ . -color, data=train_set, trControl = trainControl(method = "cv", number = 10), methodList=models_top_5)
ensemble_rf <- caretStack(ensemble_models, method="rf", metric="Accuracy", trControl=stackControl)
ensemble_rf

fit_rf <- train(quality ~ . -color, method = "rf", data = train_set, trControl = trainControl(method = "cv", number = 10))

#Tune the Worst Model - This takes forever
set.seed(1, sample.kind = "Rounding")
# Intial tuning grid. Skipping on reruns. sr_grid <- expand.grid(sigma = 2^c(-25, -20, -15,-10, -5, 0), C = 2^c(6:10))
sr_grid <- expand.grid(sigma = 1, C = 128)
fit_sr <- train(quality ~ .-color, method = "svmRadial", data = train_set, 
                trControl = trainControl(method = "cv", number = 10),
                tuneGrid = sr_grid)
#Optimal - sigma = 1 and c = 128 acc - 65% from 57%

#Tune the Second Worst Model - This takes even longer - Splitting in Half
set.seed(1, sample.kind = "Rounding")
gbm_grid <-  expand.grid(interaction.depth = c(1),
                         n.trees = (0:21)*50,
                         shrinkage = seq(.0005, .05,.005),
                         n.minobsinnode = 5)
fit_gbm <- train(quality ~ .-color, method = "gbm", data = train_set, 
                trControl = trainControl(method = "cv", number = 10),
                tuneGrid = gbm_grid)
#Second half of GBM tuning - This one is better than the first half
set.seed(1, sample.kind = "Rounding")
# Intial tuning grid. Skipping on reruns. - gbm_grid_2 <-  expand.grid(interaction.depth = c(3), n.trees = (0:21)*50, shrinkage = seq(.0005, .05,.005),  n.minobsinnode = 5)
gbm_grid_2 <- expand.grid(interaction.depth = c(3), n.trees = 1050, shrinkage = 0.0405,  n.minobsinnode = 5)
fit_gbm_2 <- train(quality ~ .-color, method = "gbm", data = train_set, 
                 trControl = trainControl(method = "cv", number = 10),
                 tuneGrid = gbm_grid_2)
#Optimal - interaction.depth = 3 and n.trees = 1050 and shrinkage = 0.0405  acc - 61% from 58.4%

#Fit Function for Evaluation of Models against Test Set
fit_function <- function(fit) {predict(fit, newdata = test_set)}

#A Function for getting the mode of the predictions
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#Run the Final Ensemble the Manual Way with an improved svmRadial and improved GBM
preds_orig <- sapply(fits_5_12, fit_function)
preds_sr <- predict(fit_sr, newdata = test_set)
preds_gbm <- predict(fit_gbm_2, newdata = test_set)
preds_all <- cbind(preds_orig, as.matrix(preds_sr), as.matrix(preds_gbm))
preds_final <- preds_all[,-c(3:4)]
preds_vote_final <- apply(preds_final, 1, getmode)
mean(preds_vote_final == test_set$quality)
#Accuracy - 70.9%

#Run the Ensemble the Manual Way
preds_2 <- preds_all[,-c(6:7)]
preds_vote_2 <- apply(preds_2, 1, getmode)
mean(preds_vote_2 == test_set$quality)
#Accuracy - 70.6%

#Run the Ensemble the Manual Way with an improved svmRadial 
preds_3 <- preds_all[,-c(4,7)]
preds_vote_3 <- apply(preds_3, 1, getmode)
mean(preds_vote_3 == test_set$quality)
#Accuracy 70.9%

#Run the initial 5 Model with 5 Factors
preds_1 <- predict(fits_top_5, newdata = test_set)
preds_vote_1 <- apply(preds_1, 1, getmode)
mean(preds_vote_1 == test_set$quality)

#Final Accuracies Dataframe & Plot
accuracies <- data.frame(method = c("Mean", "KNN", "RF", "Ens_1","Ens_2", "Ens_3", "Ens_F"), rates = c( 43.7, 47.5, 67.7, 70.6, 70.6, 70.9, 70.9))
accuracies$method <- factor(accuracies$method, levels = accuracies$method)
accuracies %>% ggplot(aes(method, rates, label = rates)) + geom_point() + 
  ggtitle("Accuracy by Method") + theme_economist() + geom_text(nudge_y = 1) + ggsave(filename = "./figure16.png")
