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
# data for lrna2
data.lrna2 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna1','cd41','cd42')]
names(data.lrna2)[2] <- 'y'
train.lrna2 <- data.lrna2[strat_index,]
test.lrna2 <- data.lrna2[-strat_index,]

#########################################################################################
para.obj <- tune(svm, y ~.,data = train.lrna2,
                 kernel = "polynomial",
                 ranges=list(gamma = c(2,1,0.1), #best (2,1,0.1)
                             degree = c(2,3,4,5,6), #best (4,5,6)
                             cost = c(0.5,0.1,0.01,0.001,0.0001)), #best(0.1,0.5)
                 tunecontrol = tune.control(sampling = "cross", cross=5,
                                            best.model = T, performances =T))

svmfit <- svm(y ~ ., data = train.lrna2, 
              kernel = 'linear',
              gamma = para.obj$best.parameters[1],
              degree = para.obj$best.parameters[2],
              cost = para.obj$best.parameters[3])
para.obj$best.parameters
# train error
ntrain <- nrow(train.lrna2)
ntest <- nrow(test.lrna2)
lrna2.pred.svm <- predict(svmfit,train.lrna2[,! colnames(train.lrna2) == 'y'])
mean.mspe.lrna2.svm <- mean((train.lrna2$y - lrna2.pred.svm)^2)
sigma.mspe.lrna2.svm <- 1/ntrain * var((train.lrna2$y - lrna2.pred.svm)^2) 
c(mean.mspe.lrna2.svm - 1.96 * sqrt(sigma.mspe.lrna2.svm), mean.mspe.lrna2.svm, mean.mspe.lrna2.svm + 1.96 * sqrt(sigma.mspe.lrna2.svm))
# 0.9095464 1.1013743 1.2932023 NOT NORM
ggplot(data.frame(cbind(lrna2.pred.svm,train.lrna2$y)), aes(x = lrna2.pred.svm, y = train.lrna2$y))+
  geom_point()  + xlim(-3,3) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'lrna2 in train set vs predicted value - SVM',x = 'Predicted value',y='True test value')

# test error
lrna2.pred.svm <- predict(svmfit,test.lrna2[,! colnames(test.lrna2) == 'y'])
mean.mspe.lrna2.svm <- mean((test.lrna2$y - lrna2.pred.svm)^2)
sigma.mspe.lrna2.svm <- 1/ntest * var((test.lrna2$y - lrna2.pred.svm)^2) 
c(mean.mspe.lrna2.svm - 1.96 * sqrt(sigma.mspe.lrna2.svm), mean.mspe.lrna2.svm, mean.mspe.lrna2.svm + 1.96 * sqrt(sigma.mspe.lrna2.svm))
# [1] 0.9099066 1.1086564 1.3074061
# 0.982927 1.322575 1.662223 NOT NORM
var(test.lrna2$y)
ggplot(data.frame(cbind(lrna2.pred.svm,test.lrna2$y)), aes(x = lrna2.pred.svm, y = test.lrna2$y))+
  geom_point()  + xlim(-2,2) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'lrna2 in test set vs predicted value - SVM',x = 'Predicted value',y='True test value')
cor(lrna2.pred.svm,test.lrna2$y)
#0.1022561 #0.07220619 NOT NORM
R2.rpart <- 1 - sum((test.lrna2$y-lrna2.pred.svm)^2)/sum((test.lrna2$y-mean(test.lrna2$y))^2)
R2.rpart
#-0.04742942 #-0.2638249 NOT NORM

# keep(list = c('dat384','prmut0','rtmut0'),sure=T)