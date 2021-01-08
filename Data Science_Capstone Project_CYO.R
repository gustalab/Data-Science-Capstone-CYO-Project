# Check all necessary libraries

if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("recorrplotadr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("pROC", repos = "http://cran.us.r-project.org")
if(!require(caTools)) install.packages("caTools", repos = "http://cran.us.r-project.org")
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
if(!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggfortify)) install.packages("ggfortify", repos = "http://cran.us.r-project.org")
if(!require(glmnet)) install.packages("glmnet", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(plotly)) install.packages("plotly", repos = "http://cran.us.r-project.org")
library(funModeling)
library(corrplot)


#################################################
#  Breast Cancer Project Code
################################################

#### Data Loading ####
# Wisconsin Breast Cancer Diagnostic Dataset
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data/version/2
# Loading the csv data file from my github account

data <- read.csv("https://raw.githubusercontent.com/gustalab/Data-Science-Capstone-CYO-Project/main/data.csv")
data$diagnosis <- as.factor(data$diagnosis)


## #In order to figure out the dataset, you can find the first rows of dataset below. There are 569 observations and 32 variables. 
head(data)

# Summary of the datasets (summary statistics of data)
summary(data)


# Proporton of diagnosises
mean(data$diagnosis == "B")    
mean(data$diagnosis == "M")
prop.table(table(data$diagnosis))

# Plotting Numerical Data
plot_num(data %>% select(-id), bins=30) 

# Plot and facet wrap density plots for each feature by diagnosis
data %>% select(-id) %>%
  gather("feature", "value", -diagnosis) %>%
  ggplot(aes(value, fill = diagnosis)) +
  geom_density(alpha = 0.5) +
  xlab("Feature values") +
  ylab("Density") +
  theme(legend.position = "top",
        axis.text.x = element_blank(), axis.text.y = element_blank(),
        legend.title=element_blank()) +
  scale_fill_discrete(labels = c("Benign", "Malignant")) +
  facet_wrap(~ feature, scales = "free", ncol = 3)



# Correlation plot
correlationMatrix <- cor(data[,3:ncol(data)])
head(round(correlationMatrix,2))

corrplot(correlationMatrix, tl.cex = 1, addrect = 8)


# Find variables that have high correlation between (>0.90)
cutoff <- 0.9
highcorrelation <- findCorrelation(correlationMatrix, cutoff=cutoff, names=TRUE)

# print the highly correlated variables
print(highcorrelation)

# Remove correlated variables
corred_data <- data %>%select(-highcorrelation)

# Number of columns after removing correlated variables
ncol(corred_data)



## Modelling

# Principal Component Analysis (PCA)

data_PCA <- prcomp(data[,3:ncol(data)], center = TRUE, scale = TRUE)
plot(data_PCA, type="b")



# Summary of data after PCA
summary(data_PCA)


# Reduce the number of variables
corred_data_PCA <- prcomp(corred_data[,3:ncol(corred_data)], center = TRUE, scale = TRUE)
corred_data_PCA
plot(corred_data_PCA, type="b")

summary(corred_data_PCA)



# PC's in the transformed data
df_PCA <- as.data.frame(corred_data_PCA$x)
head(df_PCA)
ggplot(df_PCA, aes(x=PC1, y=PC2, col=corred_data$diagnosis)) + 
  geom_point(alpha=0.5) + 
  stat_ellipse()



# Plot of PC1 and PC2
PCA1 <- ggplot(df_PCA, aes(x=PC1, fill=data$diagnosis)) + geom_density(alpha=0.25)  
PCA2 <- ggplot(df_PCA, aes(x=PC2, fill=data$diagnosis)) + geom_density(alpha=0.25)  
grid.arrange(PCA1, PCA2, ncol=2)



# Linear Discriminant Analysis (LDA)

# Data with LDA
data_LDA <- MASS::lda(diagnosis~., data = data, center = TRUE, scale = TRUE) 
data_LDA

# Visualization of LDA data frame
data_LDA_predict <- predict(data_LDA, data)$x %>% as.data.frame() %>% cbind(diagnosis=data$diagnosis)

ggplot(data_LDA_predict, aes(x=LD1, fill=diagnosis)) + 
  geom_density(alpha=0.5)


### Model creation


# Creation of the partition 80% and 20%
set.seed(1)
data2 <- cbind (diagnosis=data$diagnosis, corred_data)
data_sampling_index <- createDataPartition(data$diagnosis, times=1, p=0.8, list = FALSE)
train_data <- data2[data_sampling_index, ]
test_data <- data2[-data_sampling_index, ]


data.frame(Dataset = c("Train", "Test"),
           Benign = c(mean(train_data$diagnosis == "B"), mean(train_data$diagnosis == "B")),
           Malignant = c(mean(test_data$diagnosis == "M"), mean(test_data$diagnosis == "M")))



fitControl <- trainControl(method="cv",    
                           number = 15,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)



### Logistic Regression Model 

# Creation of Logistic Regression Model
Lreg_model <- train(diagnosis ~., 
                    train_data, 
                    method = "glm",
                    metric = "ROC",
                    preProcess = c("scale", "center"),  # in order to normalize the data
                    trControl= fitControl)
# Prediction
Lreg_model_prediction <- predict(Lreg_model, test_data)

# Confusion matrix
confusionmatrix_Lreg <- confusionMatrix(Lreg_model_prediction, test_data$diagnosis, positive = "M")
confusionmatrix_Lreg

# Plot of top important variables
plot(varImp(Lreg_model), top=10, main="Top 10 - Logistic Regression")


### K Nearest Neighbor (KNN) Model

# Creation of K Nearest Neighbor (KNN) Model
Knn_model <- train(diagnosis~.,
                   train_data,
                   method="knn",
                   metric="ROC",
                   preProcess = c('center', 'scale'),
                   tuneLength=8, #The tuneLength parameter tells the algorithm to try different default values for the main parameter
                   #In this case we used 8 default values
                   trControl=fitControl)
# Prediction
prediction_Knn <- predict(Knn_model, test_data)

# Confusion matrix        
confusionmatrix_Knn <- confusionMatrix(prediction_Knn, test_data$diagnosis, positive = "M")
confusionmatrix_Knn

# Plot of top important variables
plot(varImp(Knn_model), top=10, main="Top 10 - KNN")



### Naive Bayes Model

# Creation of Naive Bayes Model
nbayes_model <- train(diagnosis~.,
                      train_data,   
                      method="nb",
                      metric="ROC",
                      preProcess=c('center', 'scale'), #in order to normalize de data
                      trace=FALSE,
                      trControl=fitControl)

# Prediction
nbayes_prediction <- predict(nbayes_model, test_data)
# Confusion matrix
confusionmatrix_nbayes <- confusionMatrix(nbayes_prediction, test_data$diagnosis, positive = "M")
confusionmatrix_nbayes

# Plot of top important variables
plot(varImp(nbayes_model), top=10, main="Top 10 - Naive Bayes")



### Random Forest Model

# Creation of Random Forest Model
Rforest_model <- train(diagnosis~.,
                            train_data,
                            method="rf",  
                            metric="ROC",
                            preProcess = c('center', 'scale'),
                            trControl=fitControl)
# Prediction
prediction_Rforest <- predict(Rforest_model , test_data)

# Confusion matrix
confusionmatrix_Rforest <- confusionMatrix(prediction_Rforest, test_data$diagnosis, positive = "M")
confusionmatrix_Rforest

# Plot of top important variables
plot(varImp(Rforest_model), top=10, main="Top 10 - Random Forest")



### Neural Network with PCA Model

# Creation of Random Forest Model (**Neural Network with PCA model may take some time)
Nnetwork_PCA_model <- train(diagnosis~.,
                        train_data,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
# Prediction
prediction_Nnetwork_PCA <- predict(Nnetwork_PCA_model, test_data)
# Confusion matrix
confusionmatrix_Nnetwork_PCA <- confusionMatrix(prediction_Nnetwork_PCA, test_data$diagnosis, positive = "M")
confusionmatrix_Nnetwork_PCA
# Plot of top important variables
plot(varImp(Nnetwork_PCA_model), top=10, main="Top 10 - Nnetwork_PCA")


### Neural Network with LDA Model

# Creation of training set and test set with LDA modified data
train_data_lda <- data_LDA_predict[data_sampling_index, ]
test_data_lda <- data_LDA_predict[-data_sampling_index, ]

# Creation of Neural Network with LDA Mode (**Neural Network with LDA model may take some time)
Nnetwork_LDA_model <- train(diagnosis~.,
                        train_data_lda,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl=fitControl)
# Prediction
prediction_Nnetwork_LDA <- predict(Nnetwork_LDA_model, test_data_lda)
# Confusion matrix
confusionmatrix_Nnetwork_LDA <- confusionMatrix(prediction_Nnetwork_LDA, test_data_lda$diagnosis, positive = "M")
confusionmatrix_Nnetwork_LDA





# Results

# Creation of the list of all models
models_list <- list(Naive_Bayes=nbayes_model, 
                    Logistic_Reg=Lreg_model,
                    Random_Forest=Rforest_model,
                    KNN=Knn_model,
                    Neural_PCA=Nnetwork_PCA_model,
                    Neural_LDA=Nnetwork_LDA_model)                                    
models_results <- resamples(models_list)

# Print the summary of models
summary(models_results)

# Plot of the models results
bwplot(models_results, metric="ROC")

# Confusion matrix of the models
confusionmatrix_list <- list(
  Naive_Bayes=confusionmatrix_nbayes, 
  Logistic_regr=confusionmatrix_Rforest,
  KNN=confusionmatrix_Knn,
  NNetwork_PCA=confusionmatrix_Nnetwork_PCA,
  NNetwork_LDA=confusionmatrix_Nnetwork_LDA)  

confusionmatrix_list 
confusionmatrix_list_results <- sapply(confusionmatrix_list, function(x) x$byClass)
confusionmatrix_list_results


# Discussion

# Find the best result for each metric
confusionmatrix_results_max <- apply(confusionmatrix_list_results, 1, which.is.max)
output_report <- data.frame(metric=names(confusionmatrix_results_max), 
                            best_model=colnames(confusionmatrix_list_results)
                            [confusionmatrix_results_max],
                            value=mapply(function(x,y) 
                            {confusionmatrix_list_results[x,y]}, 
                            names(confusionmatrix_results_max), 
                            confusionmatrix_results_max))
rownames(output_report) <- NULL
output_report

# Appendix - Enviroment
# Print system information
print("Operating System:")
version