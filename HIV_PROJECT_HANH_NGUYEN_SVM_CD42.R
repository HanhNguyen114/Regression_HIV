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
#total_data.processed <- predict(preProcess(total_data_normalize[,! colnames(total_data_normalize) %in% c('arm')], 
 #                                          method=c('zv','nzv',"center", "scale"))
  #                              ,total_data_normalize)
total_data.processed <- predict(preProcess(total_data[,! colnames(total_data) %in% c('arm')], 
                                           method=c('zv','nzv',"center", "scale"))
                                ,total_data)
multi.hist(total_data.processed[,1:3])
############################################################################################
# index for train and test split
# Stratify data
seed <- 42
set.seed(seed)
library(caret)
strat_index <- createDataPartition(y = total_data$arm,list = FALSE,p = 0.75)

###########################################################################################
# data for cd42
data.cd42 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna1','lrna2','cd41')]
names(data.cd42)[3] <- 'y'
train.cd42 <- data.cd42[strat_index,]
test.cd42 <- data.cd42[-strat_index,]

#########################################################################################
para.obj <- tune(svm, y ~.,data = train.cd42,
                 kernel = "polynomial",
                 ranges=list(gamma = c(2,1,0.1), #best (2,1,0.1)
                             degree = c(2,3,4,5,6), #best (4,5,6)
                             cost = c(0.5,0.1,0.01,0.001,0.0001)), #best(0.1,0.5)
                 tunecontrol = tune.control(sampling = "cross", cross=5,
                                            best.model = T, performances =T))

svmfit <- svm(y ~ ., data = train.cd42, 
              kernel = 'linear',
              gamma = para.obj$best.parameters[1],
              degree = para.obj$best.parameters[2],
              cost = para.obj$best.parameters[3])
para.obj$best.parameters
# train error
ntrain <- nrow(train.cd42)
ntest <- nrow(test.cd42)
cd42.pred.svm <- predict(svmfit,train.cd42[,! colnames(train.cd42) == 'y'])
mean.mspe.cd42.svm <- mean((train.cd42$y - cd42.pred.svm)^2)
sigma.mspe.cd42.svm <- 1/ntrain * var((train.cd42$y - cd42.pred.svm)^2) 
c(mean.mspe.cd42.svm - 1.96 * sqrt(sigma.mspe.cd42.svm), mean.mspe.cd42.svm, mean.mspe.cd42.svm + 1.96 * sqrt(sigma.mspe.cd42.svm))
# 0.8308614 0.9635539 1.0962464 #0.8310954 0.9637777 1.0964600 not norm
ggplot(data.frame(cbind(cd42.pred.svm,train.cd42$y)), aes(x = cd42.pred.svm, y = train.cd42$y))+
  geom_point() + ylim(-4,6) + xlim(-0.5,0.5) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'cd42 in train set vs predicted value - SVM',x = 'Predicted value',y='True test value')

# test error
cd42.pred.svm <- predict(svmfit,test.cd42[,! colnames(test.cd42) == 'y'])
mean.mspe.cd42.svm <- mean((test.cd42$y - cd42.pred.svm)^2)
sigma.mspe.cd42.svm <- 1/ntest * var((test.cd42$y - cd42.pred.svm)^2) 
c(mean.mspe.cd42.svm - 1.96 * sqrt(sigma.mspe.cd42.svm), mean.mspe.cd42.svm, mean.mspe.cd42.svm + 1.96 * sqrt(sigma.mspe.cd42.svm))
# [1] 0.6649768 0.8608483 1.0567199 #0.6652583 0.8609834 1.0567084 not norm
var(test.cd42$y) #0.9308836
ggplot(data.frame(cbind(cd42.pred.svm,test.cd42$y)), aes(x = cd42.pred.svm, y = test.cd42$y))+
  geom_point() + ylim(-3,5) + xlim(-1,1) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'cd42 in test set vs predicted value - SVM',x = 'Predicted value',y='True test value')
cor(cd42.pred.svm,test.cd42$y)
#0.5882513 #0.5892669 not norm
R2.rpart <- 1 - sum((test.cd42$y-cd42.pred.svm)^2)/sum((test.cd42$y-mean(test.cd42$y))^2)
R2.rpart
#0.07083165 #0.07068585 not norm

# keep(list = c('dat384','prmut0','rtmut0'),sure=T)