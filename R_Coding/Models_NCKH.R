#> Install Packages <#
install.packages("ROCR")
install.packages("dplyr")
install.packages("caret")
install.packages("gmodels")
install.packages("ggplot2") # Visualize
install.packages("corrplot") # Correlation
install.packages("DMwR") #Oversampling
install.packages("RWeka") 
install.packages("C50")
install.packages("e1071")
install.packages("class")
install.packages("randomForest")
install.packages('gbm')               
install.packages('xgboost')

#> Dataset <#
# 1. Student Mat
 ### Target (y) : G_AVG

Dataset <- student_mat

# 2. Student Por
 ### Target (y): G_AVG

Dataset <- student_por

## Bonus ##
# Split to Dependent Variable (y) and Independent Variables (x)
Dataset.x <- Dataset[,!names(Dataset) %in% c("G_AVG")]
Dataset.y <- Dataset[,names(Dataset) %in% c("G_AVG")]

# 3. Sapfile1
 ### Target (y): ESP

Dataset <- Sapfile1
names(Dataset)[names(Dataset) == 'esp'] <- 'G_AVG'

table(unique(dataTrain.SMOTEed$G_AVG))
sum(is.na(as.matrix(dataTrain.SMOTEed)))

# Export Oversampling Dataset to folder
library("writexl")

write_xlsx(Dataset.SMOTEd,"D:\\Đại học\\Nghiên cứu khoa học\\10022023\\Data\\DataToSolve\\Oversampling_SMOTE_Datasets\\Smote_Sapfile1-mat.xlsx")


#> Check Dataset Information
library(tidyverse)

##1. Summary
summary(Dataset)

# Num of observations
nrow(Dataset)

# First 10 observations
head(Dataset)
  
  ##2. Check amount of NA and NaN values 
numNaN <- sum(is.nan(as.matrix(Dataset)))
numNaN
  
numNa <- sum(is.na(as.matrix(Dataset)))
numNa

##3. Plot by Histogram

#3.1. Feature before plotting
# ?pivot_longer: Pivot data from wide to long
Hist.Dataset_long <- Dataset %>%                         
                     pivot_longer(colnames(Dataset)) %>% 
                     as.data.frame()

#3.2. Plot Histogram
ggplot(Hist.Dataset_long, aes(x = value)) +
                          geom_histogram() + 
                          facet_wrap(~ name, scales = "free")


#> Correlate dataset with Pearson <#
#> 
library(corrplot)

# Split to Dependent Variable (y) and Independent Variables (x)
Dataset.x <- Dataset[,!names(Dataset) %in% c("G_AVG")]
Dataset.y <- Dataset[,names(Dataset) %in% c("G_AVG")]

corrFeature <- round(cor(y = Dataset.x, x = Dataset.y, method="pearson"),3)

#Table
table(corrFeature)

#Heatmap
corrplot(corrFeature, addCoef.col = 'black')

#> Split dataset to training set and testing set <#
library(caret)

Dataset$G_AVG <- as.factor(Dataset$G_AVG)

set.seed(3)
# Info: 66% Train / 34% Test
trainIndex <- createDataPartition(Dataset$G_AVG, p = .66, list = FALSE) #TRAIN INDEX 

dataTrain = Dataset[trainIndex, ] 
dataTest = Dataset[-trainIndex, ]


sum(is.na(dataTrain.SMOTEed)) #count the number of NA's
colSums(is.na(dataTrain.SMOTEed)) #count the NA's by column

#Transform y from numeric to factor
Dataset$G_AVG <- as.factor(Dataset$G_AVG)

#Oversampling
smote_dataTrain.SMOTEed <- as.data.frame(dataTrain)
  
table(smote_dataTrain.SMOTEed$G_AVG)
  
library(DMwR)
  
set.seed(3)
dataTrain.SMOTEed <- SMOTE(G_AVG ~ ., data = smote_dataTrain.SMOTEed, k = 3, perc.over = 1500, perc.under = 300)
###Lưu ý: Nhớ thay đổi giá trị tham số như trong báo cáo

### Xem các điểm dữ liệu trong từng class sau khi SMOTE

# SMOTE Training set
table(dataTrain.SMOTEed$G_AVG)

#Training set
table(dataTrain$G_AVG)

#Testing set
table(dataTest$G_AVG)

### Tỉ lệ của các class

#Props of SMOTE Training Set
round(proportions(table(dataTrain.SMOTEed$G_AVG)) * 100, 2)

#Props of SMOTE Training Set
round(proportions(table(dataTrain$G_AVG)) * 100, 2)

#Props of Testing Set
round(proportions(table(dataTest$G_AVG)) * 100, 2)

#Classfier Training set
numNa.test <- sum(is.na(as.matrix(dataTest)))
numNa.test

numNa.train <- sum(is.na(as.matrix(dataTrain.SMOTEed)))
numNa.train

dataTrain.SMOTEed$G_AVG <- as.factor(dataTrain.SMOTEed$G_AVG)
dataTest$G_AVG <- as.factor(dataTest$G_AVG)

#Undersampling
SCUTR.dataTrain.SMOTEed <- as.data.frame(dataTrain.SMOTEed)
# SCUTR Dataset
table(dataTrain.SMOTEed$G_AVG)
nrow(SCUTR.dataTrain.SMOTEed)
table(dataTrain.SMOTEed$G_AVG)
nrow(dataTrain.SMOTEed)

# Training set Classfiers
table(dataTrain.SMOTEed$G_AVG)

#Testing set Classfiers
table(dataTest$G_AVG)

dataTrain.SMOTEed <- undersample_tomek(
  SCUTR.dataTrain.SMOTEed,
  c(0:1),
  "G_AVG",
  20,
  tomek = "minor",
  force_m = TRUE)

dataTrain.SMOTEed$G_AVG <- as.factor(dataTrain.SMOTEed$G_AVG)
table(dataTrain.SMOTEed$G_AVG)

#Amount of Train observations
numTrain <- nrow(dataTrain.SMOTEed)
numTrain
#Amount of Test observations
numTest <- nrow(dataTest)
numTest
#> Correlate training set with Pearson <#
library(corrplot)

dataTrain.SMOTEed$G_AVG <- as.factor(dataTrain.SMOTEed$G_AVG)
corrFeature.train <- round(cor(x = as.numeric(G_AVG) ~ . , dataTrain.SMOTEed, method="pearson"),3)

#Table
table(corrFeature.train)

#Heatmap
corrplot(corrFeature.train, addCoef.col = 'black')

#> Correlate testing set with Pearson <#
library(corrplot)

corrFeature.test <- round(cor(dataTest, method="pearson"),3)

#Table
table(corrFeature.test)

#Heatmap
corrplot(corrFeature.train, addCoef.col = 'black')


### BUILD MODEL USING CV !!!

##>>> SETUP MODELS <<<##
########################

#### J48 Model #####
library(caret)

library(RWeka)


#Model
set.seed(3)
model.J48 <- J48(G_AVG ~. , dataTrain.SMOTEed,
                 control = Weka_control(), na.action =  NULL)

## model.J48.evaluate <- evaluate_Weka_classifier(model.J48,numFolds = 5)

#plot(model.J48)
table(Dataset$G_AVG)
#Summary J48 model

model.J48$pred.train <- predict(model.J48, dataTrain.SMOTEed, type = "class")
(acc.J48.train <- round(mean(model.J48$pred.train == dataTrain.SMOTEed$G_AVG), 3) * 100)

#Predict  
model.J48$pred.test <- predict(model.J48, dataTest, type = "class")
(acc.J48.test <- round(mean(model.J48$pred.test == dataTest$G_AVG), 3) * 100)

#Error
model.J48$residuals <- as.numeric(dataTest$G_AVG) - as.numeric(model.J48$pred.test)

#Check predict variable
summary(model.J48)

#Levels
levels(model.J48$pred)
levels(as.factor(dataTest$G_AVG))
#RMSE
(model.J48.RMSE <- round(sqrt(mean(model.J48$residuals^2)),4))
table(dataTest$G_AVG, model.J48$pred.test)
#Confusion Matrix
(model.J48.ConfusionMatrix <- confusionMatrix(model.J48$pred.test, as.factor(dataTest$G_AVG)))

#library(pROC)

#auc(multiclass.roc(response = dataTest$G_AVG, predictor = as.numeric(model.J48$pred.test)))

table(model.J48$pred.test, as.factor(dataTest$G_AVG))
#### C5.0 Model #####
#install.packages("C50")
library(C50)

set.seed(3)
#Model
model.C50 <- C50::C5.0(as.factor(G_AVG) ~ . , dataTrain.SMOTEed)
str(dataTrain.SMOTEed$G_AVG)
#Summary C50 model
summary(model.C50)
dataTest$G_AVG <- as.factor(dataTest$G_AVG)
#Predict    
model.C50$pred.train <- predict(model.C50, newdata = dataTrain.SMOTEed, type = "class")
(acc.C50.train <- round(mean(model.C50$pred.train == dataTrain.SMOTEed$G_AVG), 3) * 100)


model.C50$pred.test <- predict(model.C50, newdata = dataTest, type = "class")
(acc.C50.test <- round(mean(model.C50$pred.test == dataTest$G_AVG), 3) * 100)

#Error
model.C50$residuals <- as.numeric(dataTest$G_AVG) - as.numeric(model.C50$pred.test)
#Check predict variable

#RMSE
(model.C50.RMSE <- round(sqrt(mean(model.C50$residuals^2)),4))

#Confusion Matrix
(model.C50.ConfusionMatrix <- confusionMatrix(model.C50$pred.test, dataTest$G_AVG))
(model.C50.ConfusionMatrix <- confusionMatrix(model.C50$pred.train, dataTrain.SMOTEed$G_AVG))

#### K-NN Model #####
library(class)

#Get the best k value

set.seed(3)
k_total <- round(sqrt(nrow(dataTrain.SMOTEed)),0)
k_total

#Model (Using train function of caret) ### BUILD !
trainControl.KNN <- trainControl(method = "repeatedcv",
                                 number = 10,
                                 repeats = 3)

knn_fit <- train(G_AVG ~., data = dataTrain.SMOTEed, method = "knn",
                 trControl = trainControl.KNN,
                 metric = "Accuracy")
                 #tuneGrid = data.frame(k = seq(k_total - 10, k_total + 30, 3)))
                 #tuneGrid = data.frame(k = c(k_total - 4, k_total, k_total + 4)))
knn_fit$results
mean(knn_fit$results$Accuracy)

#Training set Accuracy
knn_predict.train <- predict(knn_fit, dataTrain.SMOTEed)
(acc.KNN.train <- round(mean(knn_predict.train == dataTrain.SMOTEed$G_AVG), 3) * 100)

#Testing set accuracy
knn_predict.test <- predict(knn_fit, dataTest)
(acc.KNN.test <- round(mean(knn_predict.test == dataTest$G_AVG), 3) * 100)


(cm.KNN.train <- confusionMatrix(as.factor(knn_predict.train), dataTrain.SMOTEed$G_AVG))

(cm.KNN.test <- confusionMatrix(as.factor(knn_predict.test), dataTest$G_AVG))

#Error
knn_fit$residuals <- as.numeric(dataTest$G_AVG) - as.numeric(knn_predict.test)


#RMSE
(model.KNN.RMSE <- round(sqrt(mean(knn_fit$residuals^2)),4))

#Confusion Matrix
(model.KNN.ConfusionMatrix <- confusionMatrix(knn_predict.test, as.factor(dataTest$G_AVG)))

#### Naive Bayes Model ####
library(naivebayes) # naive_bayes function
library(tidyverse)

#BUILD LAI MODEL
set.seed(3)
model.NaiveBayes <- naive_bayes(G_AVG ~ . , dataTrain.SMOTEed, laplace = 2)
#plot(model.NaiveBayes)
#Check predict var

summary(model.NaiveBayes)

model.NaiveBayes$pred.train <- predict(model.NaiveBayes, select(dataTrain.SMOTEed, -G_AVG), type = "class")
(acc.NB.train <- round(mean(model.NaiveBayes$pred.train == dataTrain.SMOTEed$G_AVG), 3) * 100)

model.NaiveBayes$pred.test <- predict(model.NaiveBayes, select(dataTest, -G_AVG), type = "class")
(acc.NB.test <- round(mean(model.NaiveBayes$pred.test == dataTest$G_AVG), 3) * 100)

table(model.NaiveBayes$pred.test, as.factor(dataTest$G_AVG))

model_KNN_residuals <- as.numeric(dataTest$G_AVG) - as.numeric(model.NaiveBayes$pred.test)

#RMSE
(model.NaiveBayes.RMSE <- round(sqrt(mean(model_KNN_residuals^2)),4))

#Underfit model NB

#Confusion Matrix
(model.NaiveBayes.ConfusionMatrix <- confusionMatrix(model.NaiveBayes$pred.test, dataTest$G_AVG))

#### SVM ####
library("e1071")

# # # Linear Kernel
  # tune.linear_SVM <- tune.svm() #Tuning
  set.seed(3)
  model.LinearSVM <- svm(as.factor(G_AVG) ~. , dataTrain.SMOTEed, kernel = "linear")
  
  #Training Acc
  model.LinearSVM$train.pred <- predict(model.LinearSVM, dataTrain.SMOTEed)
  (model.LnearSVM.train.acc <- round(mean(model.LinearSVM$train.pred == dataTrain.SMOTEed$G_AVG), 3) * 100)
  
  #Testing Acc
  model.LinearSVM$test.pred <- predict(model.LinearSVM, dataTest)
  (model.LnearSVM.test.acc <- round(mean(model.LinearSVM$test.pred == dataTest$G_AVG), 3) * 100)
  
  
  # RMSE
  model.LinearSVM$residuals <- as.numeric(dataTest$G_AVG) - as.numeric(model.LinearSVM$test.pred)
  (model.LinearSVM.RMSE <- round(sqrt(mean(model.LinearSVM$residuals^2)),4))
  
  
  # # # Polynomial Kernel
  # tune.polynomial_SVM <- tune.svm()
  
  model.PolynomialSVM <- svm(as.factor(G_AVG) ~. , dataTrain.SMOTEed, kernel = "polynomial")
  
  #Training Acc
  model.PolynomialSVM$train.pred <- predict(model.PolynomialSVM, dataTrain.SMOTEed)
  (model.PolynomialSVM.train.acc <- round(mean(model.PolynomialSVM$train.pred == dataTrain.SMOTEed$G_AVG), 3) * 100)
  
  #Testing Acc
  model.PolynomialSVM$test.pred <- predict(model.PolynomialSVM, dataTest)
  (model.PolynomialSVM.test.acc <- round(mean(model.PolynomialSVM$test.pred == dataTest$G_AVG), 3) * 100)
  
  
  # RMSE
  model.PolynomialSVM$residuals <- as.numeric(dataTest$G_AVG) - as.numeric(model.PolynomialSVM$test.pred)
  (model.PolynomialSVM.RMSE <- round(sqrt(mean(model.PolynomialSVM$residuals^2)),4))
  
  # # # RBF Kernel
  # tune.RBF_SVM <- tune.svm()
  
  model.RBF_SVM <- svm(as.factor(G_AVG) ~. , dataTrain.SMOTEed, kernel = "radial")
  
  #Training Acc
  model.RBF_SVM$train.pred <- predict(model.RBF_SVM, dataTrain.SMOTEed)
  (model.RBF_SVM.train.acc <- round(mean(model.RBF_SVM$train.pred == dataTrain.SMOTEed$G_AVG), 3) * 100)
  
  
  
  #Testing Acc
  model.RBF_SVM$test.pred <- predict(model.RBF_SVM, dataTest)
  (model.RBF_SVM.test.acc <- round(mean(model.RBF_SVM$test.pred == dataTest$G_AVG), 3) * 100)
  
  # RMSE
  model.RBF_SVM$residuals <- as.numeric(dataTest$G_AVG) - as.numeric(model.RBF_SVM$test.pred)
  (model.RBF_SVM.RMSE <- round(sqrt(mean(model.RBF_SVM$residuals^2)),4))
  
  #Accuracy of All Types
  SVM_accuracy.train <- c(model.LnearSVM.train.acc, model.PolynomialSVM.train.acc, model.RBF_SVM.train.acc)
  names(SVM_accuracy.train) <- c("Linear", "Polynomial", "RBF")
  SVM_accuracy.train
  
  SVM_accuracy.test <- c(model.LnearSVM.test.acc, model.PolynomialSVM.test.acc, model.RBF_SVM.test.acc)
  names(SVM_accuracy.test) <- c("Linear", "Polynomial", "RBF")
  SVM_accuracy.test

#### Random Forest ####
library("ranger")
library("caret")
set.seed(3)
model.RF.ranger <-  train(
                    G_AVG ~ .,
                    data = dataTrain.SMOTEed,
                    tuneLength = 1,
                    method = "ranger",
                    trControl = trainControl(
                      method = "repeatedcv", 
                      number = 10,
                      repeats = 3,
                      verboseIter = F
                    )
)
mean(model.RF.ranger$results$Accuracy)

RF_pred.train <- predict(model.RF.ranger, dataTrain.SMOTEed)
(acc.RF.train <- round(mean(RF_pred.train == dataTrain.SMOTEed$G_AVG),3) * 100)

RF_pred.test <- predict(model.RF.ranger, dataTest)
(acc.RF.test <- round(mean(RF_pred.test == dataTest$G_AVG),3) * 100)


#Summary RF model
summary(model.RF.ranger$finalModel)
summary(model.RF.ranger$results$Accuracy)

#plot(model.RF.ranger)

#Error
model.RF.ranger$residuals <- as.numeric(dataTest$G_AVG) - as.numeric(RF_pred.test)

#Check predict variable
summary(model.RF.ranger$pred)

#Levels
levels(as.factor(RF_pred.test))
levels(as.factor(dataTest$G_AVG))

#RMSE
(model.RF.RMSE <- round(sqrt(mean(model.RF.ranger$residuals^2)),4))

#Confusion Matrix
(model.RF.ConfusionMatrix <- confusionMatrix(RF_pred.test, as.factor(dataTest$G_AVG)))

#### Gradient Boosting ####
library(gbm)
set.seed(3)
#https://www.rdocumentation.org/packages/gbm/versions/2.1.8.1/topics/gbm

ctrl.GB <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 3,
                     verboseIter = FALSE) #Cross-validation

model.GBM <- train(G_AVG ~ .,
                   data = dataTrain.SMOTEed,
                   method = "gbm",
                   trControl = ctrl.GB,
                   verbose = F)
#Check model
#plot(model.GBM)

# summary(model.GBM)
# glance(model.GBM)

#Build predict

#Train pred
model.GBM$pred.train <- predict(model.GBM, dataTrain.SMOTEed)
(acc.GBM.train <- round(mean(model.GBM$pred.train == dataTrain.SMOTEed$G_AVG),3) * 100)

#Test pred
model.GBM$pred.test <- predict(model.GBM,dataTest)
(acc.GBM.test <- round(mean(model.GBM$pred.test == dataTest$G_AVG),3) * 100)

model.GBM$residuals <- as.numeric(dataTest$G_AVG) - as.numeric(model.GBM$pred.test)

# RMSE / 
(model.GBM.RMSE <- round(sqrt(mean(model.GBM$residuals^2)),4))

# Confusion Matrix /
(model.GBM.ConfusionMatrix <- confusionMatrix(model.GBM$pred.test, dataTest$G_AVG))

table(model.GBM$pred.test, dataTest$G_AVG)

#### XGBoost ####
library(tidyverse)
library(caret)
library(xgboost)

str(dataTrain.SMOTEed)
str(dataTest)

set.seed(3)
#Hyperparameters
grid_tune <- expand.grid(nrounds = c(50, 100, 150),
                         max_depth = c(2,4,6),
                         eta = 0.1,
                         gamma = 0,
                         colsample_bytree = 1,
                         min_child_weight = 1,
                         subsample = 1)

train_ctrl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 3,
                           verboseIter = T,
)

model.XGB <- train(G_AVG ~ .,
                   dataTrain.SMOTEed,
                   method = "xgbTree",
                   trControl = train_ctrl,
                   tuneGrid = grid_tune,
                   verbose = T)

#plot(model.XGB)

#Train pred
model.XGB$pred.train <- predict(model.XGB, dataTrain.SMOTEed)
(acc.XGB.train <- round(mean(model.XGB$pred.train == dataTrain.SMOTEed$G_AVG),3) * 100)

#Test pred
model.XGB$pred.test <- predict(model.XGB, dataTest)
(acc.XGB.test <- round(mean(model.XGB$pred.test == dataTest$G_AVG),3) * 100)

model.XGB$residuals <- as.numeric(dataTest$G_AVG) - as.numeric(model.XGB$pred.test)

levels(model.XGB$pred)
levels(dataTest$G_AVG)

# RMSE 
(model.XGB.RMSE <- round(sqrt(mean(model.XGB$residuals^2)),4))

#Confusion Matrix
(model.XGB.ConfusionMatrix <- confusionMatrix(model.XGB$pred.test, dataTest$G_AVG))


Train_Accuracy <- c(acc.J48.train, acc.C50.train, acc.KNN.train, acc.NB.train, SVM_accuracy.train, acc.RF.train, acc.GBM.train, acc.XGB.train)
names(Train_Accuracy) <- c("J48","C50","KNN","NB",names(SVM_accuracy.train),"RF", "GBM","XGBoost")
Train_Accuracy


Test_Accuracy <- c(acc.J48.test, acc.C50.test, acc.KNN.test, acc.NB.test, SVM_accuracy.test, acc.RF.test, acc.GBM.test, acc.XGB.test)
names(Test_Accuracy) <- c("J48","C50","KNN","NB",names(SVM_accuracy.test),"RF", "GBM","XGBoost")

RMSE_TongHop <- c(model.J48.RMSE, model.C50.RMSE, model.KNN.RMSE, model.NaiveBayes.RMSE, model.LinearSVM.RMSE, model.PolynomialSVM.RMSE, model.RBF_SVM.RMSE, model.RF.RMSE, model.GBM.RMSE, model.XGB.RMSE)
names(RMSE_TongHop) <- c("J48","C50","KNN","NB","Linear SVM", "Polynomial SVM", "Radial SVM","RF", "GBM","XGBoost")

Train_Accuracy
Test_Accuracy
RMSE_TongHop

table(dataTrain.SMOTEed$G_AVG)
table(dataTest$G_AVG)