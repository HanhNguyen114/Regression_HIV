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
total_data.processed = total_data.processed[,-c(hc)] #42 69 108 109

#boxcox
preprocessParams <- preProcess(total_data.processed[,colnames(total_data.processed) %in% c('lrna0','lrna1','lrna2','cd40','cd41','cd42')], 
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
# data for lrna1
data.lrna1 <- total_data_bc[, !colnames(total_data_bc) %in% c('lrna2','cd41','cd42')]
names(data.lrna1)[2] <- 'y'
train.lrna1 <- data.lrna1[strat_index,]
test.lrna1 <- data.lrna1[-strat_index,]
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
lm1 <- lm(y ~ .,data= train.lrna1)
#train error
lrna1.pred.lm <- predict(lm1, train.lrna1[,! colnames(train.lrna1) == "y"])
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna1.lm <- mean((test.lrna1$y - lrna1.pred.lm)^2)
sigma.mspe.lrna1.lm <- 1/ntest * var((test.lrna1$y - lrna1.pred.lm)^2) 
# 95% CI:
c(mean.mspe.lrna1.lm - 1.96 * sqrt(sigma.mspe.lrna1.lm), mean.mspe.lrna1.lm, mean.mspe.lrna1.lm + 1.96 * sqrt(sigma.mspe.lrna1.lm))
plot(lrna1.pred.lm,train.lrna1$y)
abline(a = 0, b= 1)

#test error
lrna1.pred.lm <- predict(lm1, test.lrna1[,! colnames(test.lrna1) == "y"])
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna1.lm <- mean((test.lrna1$y - lrna1.pred.lm)^2)
sigma.mspe.lrna1.lm <- 1/ntest * var((test.lrna1$y - lrna1.pred.lm)^2) 
# 95% CI:
c(mean.mspe.lrna1.lm - 1.96 * sqrt(sigma.mspe.lrna1.lm), mean.mspe.lrna1.lm, mean.mspe.lrna1.lm + 1.96 * sqrt(sigma.mspe.lrna1.lm))
plot(lrna1.pred.lm,test.lrna1$y)
abline(a = 0, b= 1)
# [1] 1.202258 1.623055 2.043852 boxcox lrna012/cd4012, all nz, center, scale, delete corr
#########################################################################################
#
################### BACKWARD FITTING ###################################################
bfit<-step(lm1,direction="backward",trace=T,k=log(dim(test.lrna1)[1]))
lm2 <- lm(y ~ lrna0 + pr72 + rt54 + rt137 + rt152 + rt156 + rt186 + rt200 + rt227 + rt233, 
          data = train.lrna1)
lrna1.pred.lm <- predict(lm1, test.lrna1[,! colnames(test.lrna1) == "y"])

# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna1.lm <- mean((test.lrna1$y - lrna1.pred.lm)^2)
sigma.mspe.lrna1.lm <- 1/ntest * var((test.lrna1$y - lrna1.pred.lm)^2) 
# 95% CI:
c(mean.mspe.lrna1.lm - 1.96 * sqrt(sigma.mspe.lrna1.lm), mean.mspe.lrna1.lm, mean.mspe.lrna1.lm + 1.96 * sqrt(sigma.mspe.lrna1.lm))
plot(lrna1.pred.lm,test.lrna1$y)
abline(a = 0, b= 1)
# [1] 1.215750 1.639698 2.063646 