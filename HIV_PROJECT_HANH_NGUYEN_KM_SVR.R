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
# data for lrna1
data.lrna1 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna2','cd41','cd42')]
data.lrna1$arm <- factor(data.lrna1$arm, labels = c(1,2,3,4,5,6), levels = c('A', 'B', 'C', 'D', 'E', 'F'))
names(data.lrna1)[2] <- 'y'
#########################################################################################
# KMEANS

set.seed(42)
wc <- NULL
for (k in 2:20){
  # k-means clustering:
  cl <- kmeans(data.lrna1,k, nstart = 10)
  # Obtain the component withinss
  wc<-c(wc, sum(cl$withinss*cl$size))
}
plot(2:20, wc, main = "k vs W(C)", xlab = "K", ylab = "W(C)")

#choose k = 3
k <- 2
set.seed(100)
km.lrna1 <- kmeans(data.lrna1, k, nstart = 5)
data.lrna1$cluster <- km.lrna1$cluster
table(data.lrna1$arm,data.lrna1$cluster)
data.lrna1.cl1 <- data.lrna1[data.lrna1$cluster == 1,]
data.lrna1.cl2 <- data.lrna1[data.lrna1$cluster == 2,]
data.lrna1 <- data.lrna1[,colnames(data.lrna1) != 'cluster']
data.lrna1.cl1 <- data.lrna1.cl1[,colnames(data.lrna1.cl1) != 'cluster']
data.lrna1.cl2 <- data.lrna1.cl2[,colnames(data.lrna1.cl2) != 'cluster']

############################################################################################
# index for train and test split
# Stratify data
seed <- 42
set.seed(seed)
library(caret)
table(data.lrna1.cl1$arm)
table(data.lrna1.cl2$arm)
strat_index1 <- createDataPartition(y = data.lrna1.cl1$arm,list = FALSE,p = 0.75)
strat_index2 <- createDataPartition(y = data.lrna1.cl2$arm,list = FALSE,p = 0.75)
###########################################################################################

train.lrna1.cl1 <- data.lrna1.cl1[strat_index1,]
test.lrna1.cl1 <- data.lrna1.cl1[-strat_index1,]
train.lrna1.cl2 <- data.lrna1.cl2[strat_index2,]
test.lrna1.cl2 <- data.lrna1.cl2[-strat_index2,]

table(train.lrna1.cl1$arm)
table(train.lrna1.cl2$arm)
#########################################################################################
para.obj1 <- tune(svm, y ~.,data = train.lrna1.cl1,
                 kernel = "polynomial",
                 ranges=list (gamma = c(2,1,0.1,0.01),
                              degree = c(4,5,6),
                              cost = c(0.1)),
                 tunecontrol = tune.control(sampling = "cross", cross = 5,
                                            nrepeat = 3, fix = 1/2,
                                            best.model = T, performances =T))
svmfit1 <- svm(y ~ ., data = train.lrna1.cl1, 
              kernel = 'polynomial',
              gamma = para.obj1$best.parameters[1],
              degree = para.obj1$best.parameters[2],
              cost = para.obj1$best.parameters[3])
para.obj2 <- tune(svm, y ~.,data = train.lrna1.cl2,
                 kernel = "polynomial",
                 ranges=list (gamma = c(2,1,0.1,0.01),
                              degree = c(4,5,6),
                              cost = c(0.1)),
                 tunecontrol = tune.control(sampling = "cross", cross = 5,
                                            nrepeat = 3, fix = 1/2,
                                            best.model = T, performances =T))
svmfit2 <- svm(y ~ ., data = train.lrna1.cl2, 
              kernel = 'polynomial',
              gamma = para.obj2$best.parameters[1],
              degree = para.obj2$best.parameters[2],
              cost = para.obj2$best.parameters[3])

#############################################################
# train error
ntrain <- nrow(train.lrna1.cl1) + nrow(train.lrna1.cl2)
ntest <- nrow(test.lrna1.cl1) + nrow(test.lrna1.cl2)

lrna1.pred.svm.cl1 <- predict(svmfit1,train.lrna1.cl1[,! colnames(train.lrna1.cl1) == 'y'])
lrna1.pred.svm.cl2 <- predict(svmfit2,train.lrna1.cl2[,! colnames(train.lrna1.cl2) == 'y'])
lrna1.pred.svm <- c(lrna1.pred.svm.cl1,lrna1.pred.svm.cl2)
train.lrna1.y <- c(train.lrna1.cl1$y,train.lrna1.cl2$y)

mean.mspe.lrna1.svm <- mean((train.lrna1.y - lrna1.pred.svm)^2)
sigma.mspe.lrna1.svm <- 1/ntrain * var((train.lrna1.y - lrna1.pred.svm)^2) 
# 95% CI:
c(mean.mspe.lrna1.svm - 1.96 * sqrt(sigma.mspe.lrna1.svm), mean.mspe.lrna1.svm, mean.mspe.lrna1.svm + 1.96 * sqrt(sigma.mspe.lrna1.svm))
plot(lrna1.pred.svm,train.lrna1.y)
lrna1.lm.train <- lm(train.lrna1.y~lrna1.pred.svm)
abline(a= 0, b=1)
#############################################################
# test error
lrna1.pred.svm.cl1 <- predict(svmfit1,test.lrna1.cl1[,! colnames(test.lrna1.cl1) == 'y'])
lrna1.pred.svm.cl2 <- predict(svmfit2,test.lrna1.cl2[,! colnames(test.lrna1.cl2) == 'y'])
lrna1.pred.svm <- c(lrna1.pred.svm.cl1,lrna1.pred.svm.cl2)
test.lrna1.y <- c(test.lrna1.cl1$y,test.lrna1.cl2$y)
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna1.svm <- mean((test.lrna1.y - lrna1.pred.svm)^2)
sigma.mspe.lrna1.svm <- 1/ntest * var((test.lrna1.y - lrna1.pred.svm)^2) 
# 95% CI:
c(mean.mspe.lrna1.svm - 1.96 * sqrt(sigma.mspe.lrna1.svm), mean.mspe.lrna1.svm, mean.mspe.lrna1.svm + 1.96 * sqrt(sigma.mspe.lrna1.svm))
# [1] 0.8577006 1.1811437 1.5, 045867 zv nzv center scale, 5cross, kernel polynomial, gamma 2,degree 5,cost 0.1
# [1] 0.857700 1.181144 1.504588, nhu tren, ngoai tru bootstrap 50,gamma 0.1,degree 5,cost 0.1
# [1] 0.9623027 1.3905732 1.8188436, linear kernal, preprocess data nhu tren, bootstrap 50,cost 0.1

plot(lrna1.pred.svm,test.lrna1.y)
lm(test.lrna1.y ~ lrna1.pred.svm)
abline(a = 0, b= 1)
sum(lrna1.pred.svm <= test.lrna1$y) # 68
sum(lrna1.pred.svm > test.lrna1$y) # more overestimate 143

var(test.lrna1.y)
