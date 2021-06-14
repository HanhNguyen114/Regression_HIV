source('HIVDATA.R')

library(dplyr)
library(ggplot2)
library(forecast)
library(caret) #train test split
library(rsample) #vfold
library(purrr) #map
library(e1071) # SVM
library(svrpath) #lasso L1 penalty 
library(tidyr)

# Preparing data
total_data <- merge(merge(dat384,prmut0, by = "patid"),rtmut0, by = "patid") #combine 3 datasets
total_data$arm <- factor(total_data$arm) # factorize arm attribute
total_data <- total_data %>% na.omit() #remove NA observations 879 - 85 = 24 obs with NA

#####################################################################################
total_data <- total_data[, ! colnames(total_data) %in% c("patid")]


# 304 - 233 = 71 constant variables

total_data_pca<- predict(preProcess(total_data[,8:ncol(total_data)], method=c('zv',"nzv", "center", "scale", "pca")), total_data)

total_data_nzv <- predict(preProcess(total_data[,! colnames(total_data) %in% c('arm')], method=c('zv',"nzv", "center", "scale"))
                          ,total_data)

total_data.processed <- predict(preProcess(total_data[,! colnames(total_data) %in% c('arm')], 
                                           method=c('zv','nzv',"center", "scale"))
                          ,total_data)
#########################################################################################
# data for cd41
data.cd41 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna2','lrna1','cd42')]
names(data.cd41)[3] <- 'y'
#########################################################################################
# RPART

set.seed(42)
library(rpart)
parameter <- rpart.control(minsplit=400, xval=10, cp=0.01)

# ***[1] 00.3155039 0.5399514 0.7643990 vary=1.261442 gamma(2,1,0.1,0.01) degree(1,2,3,4,5,6) cost(0.001,0.0001)) cor=0.1407574 bootsize=0.7
rpfit <- rpart(y ~., method="anova", control=parameter, data = data.cd41)
#printcp(rpfit)
#plot(rpfit, uniform=T, branch=0.5, compress=T, margin=.1)
#text(rpfit, all=T, use.n=F, col="blue",cex=0.8, fancy=T)
table(rpfit$where)

cl <- list()
cl[[1]] <- data.cd41[rpfit$where == 3,]
cl[[2]] <- data.cd41[rpfit$where == 4,]
cl[[3]] <- data.cd41[rpfit$where == 5,]
#cl[[4]] <- data.cd41[rpfit$where == 7,]
#[1] "frame"               "where"               "call"               
#[4] "terms"               "cptable"             "method"             
#[7] "parms"               "control"             "functions"          
#[10] "numresp"             "splits"              "csplit"             
#[13] "variable.importance" "y"                   "ordered" 
############################################################################################
# index for train and test split
# Stratify data
seed <- 42
set.seed(seed)
library(caret)
strat_index <- list()
for (i in 1:length(cl)) {
strat_index[[i]] <- createDataPartition(y = cl[[i]]$arm,list = FALSE,p = 0.75)
}
###########################################################################################
# split train test slit
train.cd41 <- list()
test.cd41 <- list()
for (i in 1:length(cl)) {
  train.cd41[[i]] <- cl[[i]][strat_index[[i]],]
  test.cd41[[i]] <- cl[[i]][-strat_index[[i]],]
}

table(train.cd41[[1]]$arm)
table(train.cd41[[2]]$arm)
table(train.cd41[[3]]$arm)
#table(train.cd41[[4]]$arm)

#########################################################################################
para.obj <- list()
svmfit <- list()
cd41.pred.svm.train <- list()
cd41.pred.svm.test <- list()
for (i in 1:length(cl)) {
  para.obj[[i]] <- tune(svm, y ~.,data = train.cd41[[i]],
                    kernel = "polynomial",
                    ranges=list (gamma = c(2,1,0.1,0.01), #best (2,1,0.1)
                                 degree = c(1,2,3,4,5,6), #best (4,5,6)
                                 cost = c(0.001,0.0001)), #best(0.1,0.5)
                    tunecontrol = tune.control(sampling = "bootstrap", nboot = 20,
                                               nrepeat = 3, boot.size = 0.7,
                                               best.model = T, performances =T))
  svmfit[[i]] <- svm(y ~ ., data = train.cd41[[i]], 
                 kernel = 'polynomial',
                 gamma = para.obj[[i]]$best.parameters[1],
                 degree = para.obj[[i]]$best.parameters[2],
                 cost = para.obj[[i]]$best.parameters[3])
  cd41.pred.svm.train[[i]] <- predict(svmfit[[i]],train.cd41[[i]][,! colnames(train.cd41[[i]]) == 'y'])
  cd41.pred.svm.test[[i]] <- predict(svmfit[[i]],test.cd41[[i]][,! colnames(test.cd41[[i]]) == 'y'])
}
#############################################################
# train error
ntrain <- 0; ntest <- 0; pred.svm.train <- NULL; train.cd41.y <- NULL
pred.svm.test <- NULL; test.cd41.y <- NULL
for (i in 1:length(cl)) {
  ntrain <- ntrain + nrow(train.cd41[[i]])
  ntest <- ntest + nrow(test.cd41[[i]])
  pred.svm.train <- c(pred.svm.train,cd41.pred.svm.train[[i]])
  train.cd41.y <- c(train.cd41.y,train.cd41[[i]]$y)
  pred.svm.test <- c(pred.svm.test,cd41.pred.svm.test[[i]])
  test.cd41.y <- c(test.cd41.y,test.cd41[[i]]$y)
}
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.cd41.svm <- mean((train.cd41.y - pred.svm.train)^2)
sigma.mspe.cd41.svm <- 1/ntrain * var((train.cd41.y - pred.svm.train)^2) 
# 95% CI:
c(mean.mspe.cd41.svm - 1.96 * sqrt(sigma.mspe.cd41.svm), mean.mspe.cd41.svm, mean.mspe.cd41.svm + 1.96 * sqrt(sigma.mspe.cd41.svm))
plot(pred.svm.train,train.cd41.y)
trainlm <- lm(train.cd41.y~ pred.svm.train - 1)
abline(a= 0, b=1)
abline(a = 0, b = trainlm$coefficients[1])
#############################################################
# test error
mean.mspe.cd41.svm <- mean((test.cd41.y - pred.svm.test)^2)
sigma.mspe.cd41.svm <- 1/ntest * var((test.cd41.y - pred.svm.test)^2) 
c(mean.mspe.cd41.svm - 1.96 * sqrt(sigma.mspe.cd41.svm), mean.mspe.cd41.svm, mean.mspe.cd41.svm + 1.96 * sqrt(sigma.mspe.cd41.svm))
plot(pred.svm.test,test.cd41.y)
testlm <- lm(test.cd41.y~ pred.svm.test - 1)
abline(a = 0, b= 1)
abline(a = 0, b = testlm$coefficients[1])
sum(pred.svm.test <= test.cd41.y) # 68
sum(pred.svm.test > test.cd41.y) # more overestimate 143

var(test.cd41.y)
ggplot(data.frame(cbind(pred.svm.test,test.cd41.y)), aes(x = pred.svm.test, y = test.cd41.y))+
  geom_point() + ylim(-3,5) + xlim(-2,2) +geom_abline(intercept = 0, slope = testlm$coefficients[1])
cor(pred.svm.test,test.cd41.y) #0.7716955

R2.rpart <- 1 - sum((test.cd41.y-pred.svm.test)^2)/sum((test.cd41.y-mean(test.cd41.y))^2)
R2.rpart #0.5698587

#############################################################
#only svr, no depart
total.train <- Reduce(rbind,train.cd41)
total.test <- Reduce(rbind,test.cd41)
para.obj.total <- tune(svm, y ~.,data = total.train,
                      kernel = "polynomial",
                      ranges=list (gamma = c(2,1,0.1,0.01), #best (2,1,0.1)
                                   degree = c(1,2,3,4,5,6), #best (4,5,6)
                                   cost = c(0.001,0.0001)), #best(0.1,0.5)
                      tunecontrol = tune.control(sampling = "bootstrap", nboot = 20,
                                                 nrepeat = 3, boot.size = 0.7,
                                                 best.model = T, performances =T))
svmfit.total <- svm(y ~ ., data = total.train, 
                   kernel = 'polynomial',
                   gamma = para.obj.total$best.parameters[1],
                   degree = para.obj.total$best.parameters[2],
                   cost = para.obj.total$best.parameters[3])
pred.svm.totr <- predict(svmfit.total,total.train[,! colnames(total.train) == 'y'])
pred.svm.tote <- predict(svmfit.total,total.test[,! colnames(total.test) == 'y'])
### train error
mean.svm <- mean((total.train$y - pred.svm.totr)^2)
sigma.svm <- 1/ntrain * var((total.train$y - pred.svm.totr)^2) 
c(mean.svm - 1.96 * sqrt(sigma.svm), mean.svm, mean.svm + 1.96 * sqrt(sigma.svm))
plot(pred.svm.totr,total.train$y)
trainlm <- lm(total.train$y~ pred.svm.totr - 1)
abline(a= 0, b=1)
abline(a = 0, b = trainlm$coefficients[1])

### test error
mean.svm <- mean((total.test$y - pred.svm.tote)^2)
sigma.svm <- 1/ntest * var((total.test$y - pred.svm.tote)^2) 
c(mean.svm - 1.96 * sqrt(sigma.svm), mean.svm, mean.svm + 1.96 * sqrt(sigma.svm))
plot(pred.svm.tote,total.test$y)
testlm <- lm(total.test$y~ pred.svm.tote - 1)
abline(a= 0, b=1)
abline(a = 0, b = testlm$coefficients[1])
var(total.test$y) #[1] 1.261442
#  test error 0.4165781 0.6663184 0.9160588

ggplot(data.frame(cbind(pred.svm.tote,y = total.test$y)), aes(x = pred.svm.tote, y = y))+
  geom_point() + ylim(-1,5) + xlim(-1,2) +geom_abline(intercept = 0, slope = testlm$coefficients[1])
cor(pred.svm.tote,total.test$y) #0.7441275

R2.0rpart <- 1 - sum((total.test$y-pred.svm.tote)^2)/sum((total.test$y-mean(total.test$y))^2)
R2.0rpart
# 0.469191