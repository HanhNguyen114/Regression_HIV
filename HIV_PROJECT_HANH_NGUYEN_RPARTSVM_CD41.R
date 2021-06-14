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
# data for cd41
data.cd41 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna1','lrna2','cd42')]
names(data.cd41)[3] <- 'y'

# after zv,nzv,center, scale: no correlated variables deleted
comat <- cor(data.cd41[, !colnames(data.cd41) %in% c('arm')])
hc = findCorrelation(comat, cutoff=0.9,exact = F) # putt any value as a "cutoff" 
hc = sort(hc) #empty
#########################################################################################
# spliting train test set
seed <- 42
set.seed(seed)
library(caret)
strat_index <- createDataPartition(y = data.cd41$arm,list = FALSE,p = 0.75)
train.cd41 <- data.cd41[strat_index,]
test.cd41 <- data.cd41[-strat_index,]
#########################################################################################
library(rpart)
parameter <- rpart.control(minsplit=100, xval=10, cp=0.00001)
rpfit <- rpart(y ~., method="anova", control=parameter, data = train.cd41)
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
  cl[[i]] <- train.cd41[rpruned$where == leaf.id[i],]
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
#pred.tree <- predict(rpruned,newdata = test.cd41[i,])
#whichleaf <- which(leaf.id == which(rpruned$frame$yval == pred.tree))
#pred.svm <- predict(svmfit[[whichleaf]],test.cd41[i,])

find.svmprediction <- function(newobs) {
  pred.tree <- predict(rpruned,newdata = newobs)
  whichleaf <- which(leaf.id == which(rpruned$frame$yval == pred.tree))
  pred.svm <- predict(svmfit[[whichleaf]],newobs)
  return(pred.svm)
}

pred.svm.vec <-NULL
for (i in 1:nrow(test.cd41)){
pred.svm.vec[i] <- find.svmprediction(test.cd41[i,])
}

#########train error
ntrain <- nrow(train.cd41); ntest <- nrow(test.cd41)
pred.svm.train.vec <- NULL; train.cd41.y <- NULL
for (i in 1:length(cl)) {
  pred.svm.train.vec <- c(pred.svm.train.vec,pred.svm.train[[i]])
  train.cd41.y <- c(train.cd41.y,cl[[i]]$y)
}
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.cd41.svm <- mean((train.cd41.y - pred.svm.train.vec)^2)
sigma.mspe.cd41.svm <- 1/ntrain * var((train.cd41.y - pred.svm.train.vec)^2) 
# 95% CI:
c(mean.mspe.cd41.svm - 1.96 * sqrt(sigma.mspe.cd41.svm), mean.mspe.cd41.svm, mean.mspe.cd41.svm + 1.96 * sqrt(sigma.mspe.cd41.svm))
#trainlm <- lm(train.cd41.y~ pred.svm.train.vec - 1)
ggplot(data.frame(cbind(pred.svm.train.vec,train.cd41$y)), aes(x = pred.svm.train.vec, y = train.cd41$y))+
  geom_point()  +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'cd41 in train set vs predicted value - RPART-SVM',x = 'Predicted value',y='True test value')
#not norm 0.07407921 0.10796034 0.14184148

#########test error
# test error
mean.mspe.cd41.svm <- mean((test.cd41$y - pred.svm.vec)^2)
sigma.mspe.cd41.svm <- 1/ntest * var((test.cd41$y - pred.svm.vec)^2) 
c(mean.mspe.cd41.svm - 1.96 * sqrt(sigma.mspe.cd41.svm), mean.mspe.cd41.svm, mean.mspe.cd41.svm + 1.96 * sqrt(sigma.mspe.cd41.svm))
#testlm <- lm(test.cd41$y~ pred.svm.vec - 1)
var(test.cd41$y)
ggplot(data.frame(cbind(pred.svm.vec,test.cd41$y)), aes(x = pred.svm.vec, y = test.cd41$y))+
  geom_point() + ylim(-3,5) + xlim(-2,4) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'cd41 in test set vs predicted value - RPART-SVM',x = 'Predicted value',y='True test value')
cor(pred.svm.vec,test.cd41$y) #not norm 0.7740633
R2.rpart <- 1 - sum((test.cd41$y-pred.svm.vec)^2)/sum((test.cd41$y-mean(test.cd41$y))^2)
R2.rpart #not norm 0.5932486

#TEST ERROR [1] 0.8659616 1.0441924 1.2224231
# not norm 0.2817039 0.3910254 0.5003468

#keep(list=c('dat384','prmut0','rtmut0'),sure=T)