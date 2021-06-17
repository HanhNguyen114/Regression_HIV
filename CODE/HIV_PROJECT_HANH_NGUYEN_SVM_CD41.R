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
# data for cd41
data.cd41 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna1','lrna2','cd42')]
names(data.cd41)[3] <- 'y'
train.cd41 <- data.cd41[strat_index,]
test.cd41 <- data.cd41[-strat_index,]

#########################################################################################
para.obj <- tune(svm, y ~.,data = train.cd41,
                 kernel = "polynomial",
                 ranges=list(gamma = c(2,1,0.1), #best (2,1,0.1)
                             degree = c(2,3,4,5,6), #best (4,5,6)
                             cost = c(0.5,0.1,0.01,0.001,0.0001)), #best(0.1,0.5)
                 tunecontrol = tune.control(sampling = "cross", cross=5,
                                            best.model = T, performances =T))

svmfit <- svm(y ~ ., data = train.cd41, 
              kernel = 'linear',
              gamma = para.obj$best.parameters[1],
              degree = para.obj$best.parameters[2],
              cost = para.obj$best.parameters[3])
para.obj$best.parameters
# train error
ntrain <- nrow(train.cd41)
ntest <- nrow(test.cd41)
cd41.pred.svm <- predict(svmfit,train.cd41[,! colnames(train.cd41) == 'y'])
mean.mspe.cd41.svm <- mean((train.cd41$y - cd41.pred.svm)^2)
sigma.mspe.cd41.svm <- 1/ntrain * var((train.cd41$y - cd41.pred.svm)^2) 
c(mean.mspe.cd41.svm - 1.96 * sqrt(sigma.mspe.cd41.svm), mean.mspe.cd41.svm, mean.mspe.cd41.svm + 1.96 * sqrt(sigma.mspe.cd41.svm))

ggplot(data.frame(cbind(cd41.pred.svm,train.cd41$y)), aes(x = cd41.pred.svm, y = train.cd41$y))+
  geom_point() + ylim(-4,6) + xlim(-2,4) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'cd41 in train set vs predicted value - SVM',x = 'Predicted value',y='True test value')
# 0.2737272 0.3305998 0.3874725 NOT NORM
# test error
cd41.pred.svm <- predict(svmfit,test.cd41[,! colnames(test.cd41) == 'y'])
mean.mspe.cd41.svm <- mean((test.cd41$y - cd41.pred.svm)^2)
sigma.mspe.cd41.svm <- 1/ntest * var((test.cd41$y - cd41.pred.svm)^2) 
c(mean.mspe.cd41.svm - 1.96 * sqrt(sigma.mspe.cd41.svm), mean.mspe.cd41.svm, mean.mspe.cd41.svm + 1.96 * sqrt(sigma.mspe.cd41.svm))
# [1] 0.9099066 1.1086564 1.3074061 # 0.2645473 0.3929662 0.5213850 NOT NORM
var(test.cd41$y) #0.9659154
ggplot(data.frame(cbind(cd41.pred.svm,test.cd41$y)), aes(x = cd41.pred.svm, y = test.cd41$y))+
  geom_point() + ylim(-4,7) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'cd41 in test set vs predicted value - SVM',x = 'Predicted value',y='True test value')
cor(cd41.pred.svm,test.cd41$y)
#0.77355 #0.7740793 NOT NORM
R2.rpart <- 1 - sum((test.cd41$y-cd41.pred.svm)^2)/sum((test.cd41$y-mean(test.cd41$y))^2)
R2.rpart
#0.5900125 #0.5912298 NOT NORM

# keep(list = c('dat384','prmut0','rtmut0'),sure=T)