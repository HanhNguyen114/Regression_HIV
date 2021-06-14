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

#zv center scale
total_data.processed <- predict(preProcess(total_data[,! colnames(total_data) %in% c('arm','patid')],
                                           method = c('zv','center','scale')),total_data)
comat <- cor(total_data.processed[,!colnames(total_data.processed) %in% c('arm')])
hc = findCorrelation(comat, cutoff=0.9,exact = F) # putt any value as a "cutoff" 
hc = sort(hc)
total_data.processed = total_data.processed[,-c(hc)]

#boxcox
preprocessParams <- preProcess(total_data.processed[,colnames(total_data.processed) %in% c('lrna0','cd41','cd41','cd40','cd41','cd42')], 
                               method=c('BoxCox'))
total_data_bc<- predict(preprocessParams, total_data.processed)
############################################################################################
# index for train and test split
# Stratify data
seed <- 42
set.seed(seed)
library(caret)
strat_index <- createDataPartition(y = total_data$arm,list = FALSE,p = 0.75)

###########################################################################################
# data for cd41
data.cd41 <- total_data_bc[, !colnames(total_data_bc) %in% c('lrna1','lrna2','cd42')]
names(data.cd41)[3] <- 'y'
train.cd41 <- data.cd41[strat_index,]
test.cd41 <- data.cd41[-strat_index,]
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
lm1 <- lm(y ~ .,data= train.cd41)
#train error
cd41.pred.lm <- predict(lm1, train.cd41[,! colnames(train.cd41) == "y"])
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.cd41.lm <- mean((test.cd41$y - cd41.pred.lm)^2)
sigma.mspe.cd41.lm <- 1/ntest * var((test.cd41$y - cd41.pred.lm)^2) 
# 95% CI:
c(mean.mspe.cd41.lm - 1.96 * sqrt(sigma.mspe.cd41.lm), mean.mspe.cd41.lm, mean.mspe.cd41.lm + 1.96 * sqrt(sigma.mspe.cd41.lm))
plot(cd41.pred.lm,train.cd41$y)
abline(a = 0, b= 1)

#test error
cd41.pred.lm <- predict(lm1, test.cd41[,! colnames(test.cd41) == "y"])
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.cd41.lm <- mean((test.cd41$y - cd41.pred.lm)^2)
sigma.mspe.cd41.lm <- 1/ntest * var((test.cd41$y - cd41.pred.lm)^2) 
# 95% CI:
c(mean.mspe.cd41.lm - 1.96 * sqrt(sigma.mspe.cd41.lm), mean.mspe.cd41.lm, mean.mspe.cd41.lm + 1.96 * sqrt(sigma.mspe.cd41.lm))
plot(cd41.pred.lm,test.cd41$y)
abline(a = 0, b= 1)
# [1] 0.4895530 0.7089056 0.9282582 boxcox lrna012/cd4012, all nz, center, scale, delete corr
#########################################################################################
#
################### BACKWARD FITTING ###################################################
bfit<-step(lm1,direction="backward",trace=T,k=log(dim(test.cd41)[1]))
lm2 <- lm(y ~ lrna0 + cd40 + pr15 + pr17 + pr64 + rt39 + rt56 + rt99 + 
            rt136 + rt156 + rt178 + rt203 + rt215, 
          data = train.cd41)
cd41.pred.lm <- predict(lm1, test.cd41[,! colnames(test.cd41) == "y"])

# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.cd41.lm <- mean((test.cd41$y - cd41.pred.lm)^2)
sigma.mspe.cd41.lm <- 1/ntest * var((test.cd41$y - cd41.pred.lm)^2) 
# 95% CI:
c(mean.mspe.cd41.lm - 1.96 * sqrt(sigma.mspe.cd41.lm), mean.mspe.cd41.lm, mean.mspe.cd41.lm + 1.96 * sqrt(sigma.mspe.cd41.lm))
# [1] 0.4895530 0.7089056 0.9282582
plot(cd41.pred.lm,test.cd41$y)
abline(a = 0, b= 1)

