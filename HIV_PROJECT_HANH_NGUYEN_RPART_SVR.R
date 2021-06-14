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
total_data_normalize<- predict(preProcess(total_data[,1:3], method=c('YeoJohnson')), total_data)

library(psych)
total_data.processed <- predict(preProcess(total_data_normalize[,! colnames(total_data_normalize) %in% c('arm')], 
                                           method=c('zv','nzv',"center", "scale"))
                                ,total_data_normalize)
multi.hist(total_data.processed[,1:3])
#########################################################################################
# data for lrna1
data.lrna1 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna2','cd41','cd42')]
names(data.lrna1)[2] <- 'y'
#########################################################################################
# RPART

set.seed(42)
library(rpart)
parameter <- rpart.control(minsplit=300, xval=10, cp=0.01)
# [1] 0.6864215 0.9179665 1.1495115 vary=0.9049796 minsplit=300, xval=10, cp=0.01 with nzv cor=0.143114 bootsize=0.9
# [1] 0.6864215 0.9179665 1.1495115 vary=0.9049796 minsplit=300, xval=10, cp=0.01 with nzv cor=0.143114 bootsize=0.8, gamma(2,1,0.1) degree(4,5,6) cost(0.1,0.5)
# ***[1] 0.6511896 0.9120503 1.1729109 vary=0.9049796 minsplit=300, xval=10, cp=0.01 with nzv cor=0.1021866 bootsize=0.7, gamma = c(1,0.1,0.01) degree = c(3,4),cost = c(0.01)
# [1] 0.6923323 0.9828739 1.2734155 vary=1.00568 minsplit=300, xval=10, cp=0.001 with nzv cor=0.1472532 bootsize=0.8
# [1] 0.6697973 0.9123655 1.1549338 vary=0.9049796 gamma(2,1,0.1,0.01) degree(1,2,3,4,5,6) cost(0.001,0.0001)) cor=0.1407574 bootsize=0.7
# 0.7741628 0.9327347 1.0913065 minsplit=300, xval=10, cp=0.01 with norm lrna amma(2,1,0.1,0.01) degree(1,2,3,4,5,6) cost(0.001,0.0001)) bootsize0.8

#[1] 0.7519329 0.9085627 1.0651924 minsplit=300, xval=10, cp=0.01 YeoJohnson lrna amma(2,1,0.1,0.01) degree(1,2,3,4,5,6) cost(0.001,0.0001)) bootsize0.7
rpfit <- rpart(y ~., method="anova", control=parameter, data = data.lrna1)
#printcp(rpfit)
#plot(rpfit, uniform=T, branch=0.5, compress=T, margin=.1)
#text(rpfit, all=T, use.n=F, col="blue",cex=0.8, fancy=T)
table(rpfit$where)

cl <- list()
cl[[1]] <- data.lrna1[rpfit$where == 2,]
cl[[2]] <- data.lrna1[rpfit$where == 4,]
cl[[3]] <- data.lrna1[rpfit$where == 5,]
#cl[[4]] <- data.lrna1[rpfit$where == 7,]
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
train.lrna1 <- list()
test.lrna1 <- list()
for (i in 1:length(cl)) {
  train.lrna1[[i]] <- cl[[i]][strat_index[[i]],]
  test.lrna1[[i]] <- cl[[i]][-strat_index[[i]],]
}

table(train.lrna1[[1]]$arm)
table(train.lrna1[[2]]$arm)
table(train.lrna1[[3]]$arm)
#table(train.lrna1[[4]]$arm)

#########################################################################################
para.obj <- list()
svmfit <- list()
lrna1.pred.svm.train <- list()
lrna1.pred.svm.test <- list()
for (i in 1:length(cl)) {
  para.obj[[i]] <- tune(svm, y ~.,data = train.lrna1[[i]],
                    kernel = "polynomial",
                    ranges=list  (gamma = c(2,1,0.1,0.01), #best (2,1,0.1)
                                  degree = c(1,2,3,4,5,6), #best (4,5,6)
                                  cost = c(0.001,0.0001)), #best(0.1,0.5)
                    tunecontrol = tune.control(sampling = "bootstrap", nboot = 20,
                                               nrepeat = 3, boot.size = 0.7,
                                               best.model = T, performances =T))
  svmfit[[i]] <- svm(y ~ ., data = train.lrna1[[i]], 
                 kernel = 'polynomial',
                 gamma = para.obj[[i]]$best.parameters[1],
                 degree = para.obj[[i]]$best.parameters[2],
                 cost = para.obj[[i]]$best.parameters[3])
  lrna1.pred.svm.train[[i]] <- predict(svmfit[[i]],train.lrna1[[i]][,! colnames(train.lrna1[[i]]) == 'y'])
  lrna1.pred.svm.test[[i]] <- predict(svmfit[[i]],test.lrna1[[i]][,! colnames(test.lrna1[[i]]) == 'y'])
}
#############################################################
# train error
ntrain <- 0; ntest <- 0; pred.svm.train <- NULL; train.lrna1.y <- NULL
pred.svm.test <- NULL; test.lrna1.y <- NULL
for (i in 1:length(cl)) {
  ntrain <- ntrain + nrow(train.lrna1[[i]])
  ntest <- ntest + nrow(test.lrna1[[i]])
  pred.svm.train <- c(pred.svm.train,lrna1.pred.svm.train[[i]])
  train.lrna1.y <- c(train.lrna1.y,train.lrna1[[i]]$y)
  pred.svm.test <- c(pred.svm.test,lrna1.pred.svm.test[[i]])
  test.lrna1.y <- c(test.lrna1.y,test.lrna1[[i]]$y)
}
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna1.svm <- mean((train.lrna1.y - pred.svm.train)^2)
sigma.mspe.lrna1.svm <- 1/ntrain * var((train.lrna1.y - pred.svm.train)^2) 
# 95% CI:
c(mean.mspe.lrna1.svm - 1.96 * sqrt(sigma.mspe.lrna1.svm), mean.mspe.lrna1.svm, mean.mspe.lrna1.svm + 1.96 * sqrt(sigma.mspe.lrna1.svm))
plot(pred.svm.train,train.lrna1.y)
trainlm <- lm(train.lrna1.y~ pred.svm.train - 1)
abline(a= 0, b=1)
abline(a = 0, b = trainlm$coefficients[1])
#############################################################
# test error
mean.mspe.lrna1.svm <- mean((test.lrna1.y - pred.svm.test)^2)
sigma.mspe.lrna1.svm <- 1/ntest * var((test.lrna1.y - pred.svm.test)^2) 
c(mean.mspe.lrna1.svm - 1.96 * sqrt(sigma.mspe.lrna1.svm), mean.mspe.lrna1.svm, mean.mspe.lrna1.svm + 1.96 * sqrt(sigma.mspe.lrna1.svm))
plot(pred.svm.test,test.lrna1.y)
testlm <- lm(test.lrna1.y~ pred.svm.test - 1)
abline(a = 0, b= 1)
abline(a = 0, b = testlm$coefficients[1])
sum(pred.svm.test <= test.lrna1.y) # 68
sum(pred.svm.test > test.lrna1.y) # more overestimate 143

var(test.lrna1.y)
ggplot(data.frame(cbind(pred.svm.test,test.lrna1.y)), aes(x = pred.svm.test, y = test.lrna1.y))+
  geom_point() + ylim(-2,5) + xlim(-1,2) +geom_abline(intercept = 0, slope = testlm$coefficients[1])
cor(pred.svm.test,test.lrna1.y)

R2.rpart <- 1 - sum((test.lrna1.y-pred.svm.test)^2)/sum((test.lrna1.y-mean(test.lrna1.y))^2)
R2.rpart
#-0.01298522 or  0.007004196 with norm lrna1; 0.07312729 yeojs lrna1
#############################################################
#only svr, no depart
total.train <- Reduce(rbind,train.lrna1)
total.test <- Reduce(rbind,test.lrna1)
para.obj.total <- tune(svm, y ~.,data = total.train,
                      kernel = "polynomial",  
                      ranges=list (gamma = c(2,1,0.1,0.01), #best (2,1,0.1)
                                   degree = c(1,2,3,4,5,6), #best (4,5,6)
                                   cost = c(0.001,0.0001)), #best(0.1,0.5)
                      tunecontrol = tune.control(sampling = "bootstrap", nboot = 10,
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
var(total.test$y) #[1] 0.9049796
# [1] 0.7467843 0.9803498 1.2139152 test error

ggplot(data.frame(cbind(pred.svm.tote,y = total.test$y)), aes(x = pred.svm.tote, y = y))+
  geom_point() + ylim(-1,5) + xlim(-1,2) +geom_abline(intercept = 0, slope = testlm$coefficients[1])
cor(pred.svm.tote,total.test$y) #0.03687526

R2.0rpart <- 1 - sum((total.test$y-pred.svm.tote)^2)/sum((total.test$y-mean(total.test$y))^2)
R2.0rpart
# -0.08846704
