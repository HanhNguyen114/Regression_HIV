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
preprocessParams <- preProcess(total_data.processed[,colnames(total_data.processed) %in% c('lrna0','lrna2','lrna2','cd40','cd41','cd42')], 
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
# data for lrna2
data.lrna2 <- total_data_bc[, !colnames(total_data_bc) %in% c('lrna1','cd41','cd42')]
names(data.lrna2)[2] <- 'y'
train.lrna2 <- data.lrna2[strat_index,]
test.lrna2 <- data.lrna2[-strat_index,]
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
lm1 <- lm(y ~ .,data= train.lrna2)
#train error
lrna2.pred.lm <- predict(lm1, train.lrna2[,! colnames(train.lrna2) == "y"])
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna2.lm <- mean((test.lrna2$y - lrna2.pred.lm)^2)
sigma.mspe.lrna2.lm <- 1/ntest * var((test.lrna2$y - lrna2.pred.lm)^2) 
# 95% CI:
c(mean.mspe.lrna2.lm - 1.96 * sqrt(sigma.mspe.lrna2.lm), mean.mspe.lrna2.lm, mean.mspe.lrna2.lm + 1.96 * sqrt(sigma.mspe.lrna2.lm))
plot(lrna2.pred.lm,train.lrna2$y)
abline(a = 0, b= 1)

#test error
lrna2.pred.lm <- predict(lm1, test.lrna2[,! colnames(test.lrna2) == "y"])
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna2.lm <- mean((test.lrna2$y - lrna2.pred.lm)^2)
sigma.mspe.lrna2.lm <- 1/ntest * var((test.lrna2$y - lrna2.pred.lm)^2) 
# 95% CI:
c(mean.mspe.lrna2.lm - 1.96 * sqrt(sigma.mspe.lrna2.lm), mean.mspe.lrna2.lm, mean.mspe.lrna2.lm + 1.96 * sqrt(sigma.mspe.lrna2.lm))
plot(lrna2.pred.lm,test.lrna2$y)
abline(a = 0, b= 1)
# [1] 1.286855 1.772347 2.257839 boxcox lrna012/cd4012, all nz, center, scale, delete corr
#########################################################################################
#
################### BACKWARD FITTING ###################################################
bfit<-step(lm1,direction="backward",trace=T,k=log(dim(test.lrna2)[1]))
lm2 <- lm(y ~ pr62 + pr67 + pr70 + rt53 + rt69 + rt136 + rt137 + rt144 + rt156 + rt172 + rt186 + rt208 + rt227 + rt233, 
          data = train.lrna2)
lrna2.pred.lm <- predict(lm1, test.lrna2[,! colnames(test.lrna2) == "y"])

# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna2.lm <- mean((test.lrna2$y - lrna2.pred.lm)^2)
sigma.mspe.lrna2.lm <- 1/ntest * var((test.lrna2$y - lrna2.pred.lm)^2) 
# 95% CI:
c(mean.mspe.lrna2.lm - 1.96 * sqrt(sigma.mspe.lrna2.lm), mean.mspe.lrna2.lm, mean.mspe.lrna2.lm + 1.96 * sqrt(sigma.mspe.lrna2.lm))
plot(lrna2.pred.lm,test.lrna2$y)
abline(a = 0, b= 1)
#[1] 1.286855 1.772347 2.257839