library(caret)
library(glmnet)

dataset<- read.csv(file = "census_income_learn.csv", header = FALSE, stringsAsFactors=TRUE)
set.seed(1011)

levels(dataset$V42) <- c("0","1")

# get the dataset with new features
dataset_new <- prepare_data(dataset, "train")


#### Duplicate some rows for oversampling (Deal with Imbalanced class distribution)

rows_rich <- which(dataset_new$class == "1")
rows_poor <- which(dataset_new$class == "0")
# create random numbers 10 times of rich rows length
random_number <- sample(1:length(rows_rich), (length(rows_rich) * 10), replace=T)
dataset_rich <- dataset_new[rows_rich,]
oversampled_rows <- dataset_rich[random_number,]

dataset_oversampled <- rbind(dataset_new, oversampled_rows)
train_data <- dataset_oversampled[sample(nrow(dataset_oversampled)), ]


################# Feature Selection by Lasso  ######

m2<- data.matrix(train_data[,1:244])
# Find the best lambda value by cross validation
cv.glmmod <- cv.glmnet(m2, train_data[,"class"], alpha=1, family="binomial", type.measure = "deviance")
best_lambda <- cv.glmmod$lambda.min
#best_lambda <- 8.600742e-05

res.lasso = glmnet(m2, train_data$class, alpha =1, family="binomial")
pr<-predict(res.lasso, type ="coefficients", s= c(best_lambda))
nonzero_coef<-predict(res.lasso, type ="nonzero", s= c(best_lambda))
length(nonzero_coef$X1) # 219 features have non zero coefficients
new_col<-colnames(train_data)[nonzero_coef$X1]

#new training data
train_data_new <- train_data[,new_col]
train_data_new <- cbind(train_data_new, train_data$class)
colnames(train_data_new)[220] <- "class"


# features with high lasso coefficients 

co <- coef(res.lasso,s=best_lambda,exact=TRUE)
sorted_co <- sort(abs(co@x), decreasing = TRUE , index.return = TRUE)
imp_features <- colnames(train_data_new)[sorted_co$ix[1:50]] #first 50 important features

###### Create model (Logistic Regression) ######


## Evaluation of model with cross validation 

accuracy_list <- c()
folds <- cut(seq(1,nrow(train_data_new)),breaks=5,labels=FALSE)
for(i in 1:5){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData_cv <- train_data_new[testIndexes, ]
  trainData_cv <- train_data_new[-testIndexes, ]
  model_cv <- glm( trainData_cv$class ~ ., data = trainData_cv, family = "binomial")
  pred_cv <- predict(model_cv, newdata = testData_cv, type = "response")
  fitted.results <- ifelse(pred_cv > 0.5,1,0)
  tbl <- table(fitted.results, testData_cv$class)
  acc <- sum(diag(tbl))/sum(tbl)
  accuracy_list <- c(accuracy_list, acc )
}

cv_accuracy <- mean(accuracy_list) # 0.86 cross valdation accuracy


### Evaluation of model on real test data
#prepare test data
test_data <- read.csv(file = "us_census_full//census_income_test.csv" , header = FALSE, stringsAsFactors=TRUE)
test_class <- test_data$V42
test_dataset_new <-prepare_data(test_data, "test")
test_dataset_new <- test_dataset_new[,new_col]

# Create the model
model <- glm( train_data_new$class ~ ., data = train_data_new, family = "binomial")
pred <- predict(model, newdata = test_dataset_new, type = "response")
fitted.results <- ifelse(pred > 0.5,1,0)

tbl <- table(fitted.results, test_class)
sum(diag(tbl))/sum(tbl)   ##### 87% accuracy
confusionMatrix(fitted.results, test_class)

############ Model 2: Random Forest ###########
library(randomForest)

# Evaluation of model with cross validation
accuracy_list <- c()
folds <- cut(seq(1,nrow(train_data_new)),breaks=5,labels=FALSE)
for(i in 1:5){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData_cv <- train_data_new[testIndexes, ]
  trainData_cv <- train_data_new[-testIndexes, ]
  model_cv <- randomForest(trainData_cv[,220] ~. , data=trainData_cv,ntree=100, mtry=5) 
  pred_cv <- predict(model_cv, newdata = testData_cv)
  tbl <- table(pred_cv, testData_cv[,220])
  acc <- sum(diag(tbl))/sum(tbl)
  print(acc)
  accuracy_list <- c(accuracy_list, acc )
}

cv_accuracy <- mean(accuracy_list) # average accuracy is 0.97
print(cv_accuracy)

## Model on Real Test Dataset
model <- randomForest(train_data_new[,220] ~. , data=train_data_new,ntree=100, mtry=5) 
#colnames(test_dataset_new) [220]<-"class"
pred <- predict(model_cv, newdata = test_dataset_new)
tbl<-table(pred, test_data[,220])
acc <- sum(diag(tbl))/sum(tbl) #98%
print(acc)
confusionMatrix(pred, test_class)



###### Prepare Data Function ##########
prepare_data <- function(dataset, str){
  
  #change education feature to ordered version
  dataset$V5 <- factor(dataset$V5, levels = levels(dataset$V5)[c(11,14,4:7,1:3,13,17,9,8,10,16,15,12)])
  rdata = factor(dataset$V5,labels=c(1:17))
  dataset$V5 <-  as.numeric(levels(rdata))[rdata] 
  class <- dataset$V42
  # remove some features
  dataset <- dataset[,-c(3,4, 23,33,34,42)] 
  
  numeric_variables <- sapply(dataset, is.numeric)
  dataset_num <- dataset[, numeric_variables]
  dataset_factor <- dataset[,!numeric_variables]
  
  #convert factors to binary features
  binary_features <- model.matrix( ~ . -1, data=dataset_factor )
  if(str == "train")
  {
    new_dataset <- cbind(dataset_num,as.data.frame(binary_features), class)
  }else{
    new_dataset <- cbind(dataset_num,as.data.frame(binary_features)) 
  }
  
  return(new_dataset)
}
