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
  #                        ,total_data_normalize)
total_data.processed <- predict(preProcess(total_data[,! colnames(total_data) %in% c('arm')], 
                                           method=c('zv','nzv',"center", "scale"))
                                ,total_data)
multi.hist(total_data.processed[,1:3])
#########################################################################################
# data for lrna1
data.lrna1 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna2','cd41','cd42')]
names(data.lrna1)[2] <- 'y'

# after zv,nzv,center, scale: no correlated variables deleted
comat <- cor(data.lrna1[, !colnames(data.lrna1) %in% c('arm')])
hc = findCorrelation(comat, cutoff=0.9,exact = F) # putt any value as a "cutoff" 
hc = sort(hc) #empty
#########################################################################################
# spliting train test set
seed <- 42
set.seed(seed)
library(caret)
strat_index <- createDataPartition(y = data.lrna1$arm,list = FALSE,p = 0.75)
train.lrna1 <- data.lrna1[strat_index,]
test.lrna1 <- data.lrna1[-strat_index,]
#########################################################################################
library(rpart)
parameter <- rpart.control(minsplit=100, xval=10, cp=0.00001)
rpfit <- rpart(y ~., method="anova", control=parameter, data = train.lrna1)
table(rpfit$where)
rpruned <- prune(rpfit, cp = rpfit$cptable[which.min(rpfit$cptable[,"xerror"]),"CP"])
#plot(rpruned, uniform=T, branch=0.5, compress=T, margin=.1)
#text(rpruned, all=T, use.n=F, col="blue",cex=0.8, fancy=T)
table(rpruned$where)
leaf.id <- unique(rpruned$where) #id of leaf after pruning
k <- length(leaf.id) #number of terminal node

#### Make a list of data cluster in each leaf
cl <- list()
for (i in 1:k) {
  cl[[i]] <- train.lrna1[rpruned$where == leaf.id[i],]
}
 
#### SVR for each local leaf
para.obj <- list()
svmfit <- list()
pred.svm.train <- list()

for (i in 1:length(cl)) {
  para.obj[[i]] <- tune(svm, y ~.,data = cl[[i]],
                        kernel = "polynomial",
                        ranges=list(gamma = c(2,1,0.1,0.01), #best (2,1,0.1)
                                    degree = c(1,2,3,4,5,6), #best (4,5,6)
                                    cost = c(0.001,0.0001)), #best(0.1,0.5)
                        tunecontrol = tune.control(sampling = "bootstrap", nboot = 20,
                                                   nrepeat = 3, boot.size = 0.7,
                                                   best.model = T, performances =T))
  svmfit[[i]] <- svm(y ~ ., data = cl[[i]],
                     kernel = 'polynomial',
                     gamma = para.obj[[i]]$best.parameters[1],
                     degree = para.obj[[i]]$best.parameters[2],
                     cost = para.obj[[i]]$best.parameters[3])
  pred.svm.train[[i]] <- predict(svmfit[[i]],cl[[i]][,! colnames(cl[[i]]) == 'y'])
}

###### prediction test set using tree and svm
#i = 1
#pred.tree <- predict(rpruned,newdata = test.lrna1[i,])
#whichleaf <- which(leaf.id == which(rpruned$frame$yval == pred.tree))
#pred.svm <- predict(svmfit[[whichleaf]],test.lrna1[i,])

find.svmprediction <- function(newobs) {
  pred.tree <- predict(rpruned,newdata = newobs)
  whichleaf <- which(leaf.id == which(rpruned$frame$yval == pred.tree))
  pred.svm <- predict(svmfit[[whichleaf]],newobs)
  return(pred.svm)
}

pred.svm.vec <-NULL
for (i in 1:nrow(test.lrna1)){
pred.svm.vec[i] <- find.svmprediction(test.lrna1[i,])
}

#########train error
ntrain <- nrow(train.lrna1); ntest <- nrow(test.lrna1)
pred.svm.train.vec <- NULL; train.lrna1.y <- NULL
for (i in 1:length(cl)) {
  pred.svm.train.vec <- c(pred.svm.train.vec,pred.svm.train[[i]])
  train.lrna1.y <- c(train.lrna1.y,cl[[i]]$y)
}
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna1.svm <- mean((train.lrna1.y - pred.svm.train.vec)^2)
sigma.mspe.lrna1.svm <- 1/ntrain * var((train.lrna1.y - pred.svm.train.vec)^2) 
# 95% CI:
c(mean.mspe.lrna1.svm - 1.96 * sqrt(sigma.mspe.lrna1.svm), mean.mspe.lrna1.svm, mean.mspe.lrna1.svm + 1.96 * sqrt(sigma.mspe.lrna1.svm))
#trainlm <- lm(train.lrna1.y~ pred.svm.train.vec - 1)
ggplot(data.frame(cbind(pred.svm.train.vec,train.lrna1$y)), aes(x = pred.svm.train.vec, y = train.lrna1$y))+
  geom_point() + ylim(-2,5) + xlim(-1,4) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'lrna1 in train set vs predicted value - RPART-SVM',x = 'Predicted value',y='True test value')
# 0.009798019 0.039827635 0.069857250 NOT NORM
#########test error
# test error
mean.mspe.lrna1.svm <- mean((test.lrna1$y - pred.svm.vec)^2)
sigma.mspe.lrna1.svm <- 1/ntest * var((test.lrna1$y - pred.svm.vec)^2) 
c(mean.mspe.lrna1.svm - 1.96 * sqrt(sigma.mspe.lrna1.svm), mean.mspe.lrna1.svm, mean.mspe.lrna1.svm + 1.96 * sqrt(sigma.mspe.lrna1.svm))
#testlm <- lm(test.lrna1$y~ pred.svm.vec - 1)
var(test.lrna1$y)
ggplot(data.frame(cbind(pred.svm.vec,test.lrna1$y)), aes(x = pred.svm.vec, y = test.lrna1$y))+
  geom_point()  + xlim(-1,2) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'lrna1 in test set vs predicted value - RPART-SVM',x = 'Predicted value',y='True test value')
cor(pred.svm.vec,test.lrna1$y) #0.104464 NOT NORM
R2.rpart <- 1 - sum((test.lrna1$y-pred.svm.vec)^2)/sum((test.lrna1$y-mean(test.lrna1$y))^2)
R2.rpart #-0.03169664 NOT NORM

#TEST ERROR [1] 0.8659616 1.0441924 1.2224231
# 0.8614556 1.1930941 1.5247326 NOT NORM

#keep(list=c('dat384','prmut0','rtmut0'),sure=T)