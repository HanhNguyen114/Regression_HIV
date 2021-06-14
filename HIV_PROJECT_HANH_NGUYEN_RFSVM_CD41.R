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
library(rpart)

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
# making 100 bootstrap sample
nboot <- 100
set.seed(seed)
train.id <- dimnames(train.cd41)[[1]]
boot.id <- replicate(nboot,sample(train.id, replace = TRUE))
boot.train <- list()
for (i in 1:nboot) {
  boot.train[[i]] <- train.cd41[boot.id[,i],]
}
#########################################################################################
rpruned <- list()
subn <- 50
cl <- list() # each element of this list is a sublist, the sublist if data of each cluster at terminal node of rpruned tree
para.obj <- list()
svmfit <- list()
pred.svm.train <- list()
pred.svm.vec <- list() #list of predicted values vectors of test set, vector[[i]] based on boot.train[[i]]
mean.sq.error <- vector()
for (i in 1:nboot) {
  #choose subset of subn variables for regression tree
  set.seed(seed)
  parameter <- rpart.control(minsplit=200, xval=10, cp=0.00001)
  subset.var <- sample(dimnames(boot.train[[i]][, colnames(boot.train[[i]]) != 'y'])[[2]])[1:subn]
  temp.data <- boot.train[[i]][, c(subset.var,'y')]
  ############ fit and prune a tree
  rpfit <- rpart(y ~., method="anova", control=parameter, data = temp.data)
  rpruned[[i]] <- prune(rpfit, cp = rpfit$cptable[which.min(rpfit$cptable[,"xerror"]),"CP"])
  print(i)
  print(table(rpruned[[i]]$where))
  leaf.id <- unique(rpruned[[i]]$where)
  k <- length(leaf.id)
  ############# clusters at terminal nodes of the tree based on temp.data
  cl[[i]] <- list()
  for (j in 1:k) {
    cl[[i]][[j]] <- temp.data[rpruned[[i]]$where == leaf.id[j],]
  }
  ########## fit SVR for each cluster
  para.obj[[i]] <- list()
  svmfit[[i]] <- list()
  pred.svm.train[[i]] <- list()
  pred.svm.train.vec <- NULL
  for (j in 1:length(cl[[i]])) {
    para.obj[[i]][[j]] <- tune(svm, y ~.,data = cl[[i]][[j]],
                          kernel = "polynomial",
                          ranges=list(gamma = c(2,1,0.1), #best (2,1,0.1)
                                      degree = c(2,3,4,5,6), #best (4,5,6)
                                      cost = c(0.1,0.01,0.001)), #best(0.1,0.5)
                          tunecontrol = tune.control(sampling = "cross", cross = 5,
                                                     best.model = T, performances =T))
    svmfit[[i]][[j]] <- svm(y ~ ., data = cl[[i]][[j]],
                       kernel = 'polynomial',
                       gamma = para.obj[[i]][[j]]$best.parameters[1],
                       degree = para.obj[[i]][[j]]$best.parameters[2],
                       cost = para.obj[[i]][[j]]$best.parameters[3])
    pred.svm.train[[i]][[j]] <- predict(svmfit[[i]][[j]],cl[[i]][[j]][,! colnames(cl[[i]][[j]]) == 'y'])
    pred.svm.train.vec <- c( pred.svm.train.vec, pred.svm.train[[i]][[j]])
  }
  mean.sq.error[i] <- mean((temp.data$y - pred.svm.train.vec)^2)
  pred.svm.vec[[i]] <- vector()
  for (j in 1:nrow(test.cd41)){
    newobs <- test.cd41[j,]
    pred.tree <- predict(rpruned[[i]],newdata = newobs)
    whichleaf <- which(leaf.id == which(rpruned[[i]]$frame$yval == pred.tree))
    pred.svm.vec[[i]][j] <- predict(svmfit[[i]][[whichleaf]],newobs)
  }
  print(pred.svm.vec[[i]])
}

pred.svm.test.df <- as.data.frame(do.call(cbind, pred.svm.vec))
pred.svm.test <- apply(pred.svm.test.df,1, mean)

#########train error
ntrain <- nrow(train.cd41); ntest <- nrow(test.cd41)
mean.train.error <- mean(mean.sq.error)
mean.train.error #1.925316 subn=50
# not norm subn=50 1.925321
#########test error
# test error
mean.mspe.cd41.svm <- mean((test.cd41$y - pred.svm.test)^2)
sigma.mspe.cd41.svm <- 1/ntest * var((test.cd41$y - pred.svm.test)^2) 
c(mean.mspe.cd41.svm - 1.96 * sqrt(sigma.mspe.cd41.svm), mean.mspe.cd41.svm, mean.mspe.cd41.svm + 1.96 * sqrt(sigma.mspe.cd41.svm))
# 0.2478698 0.3578774 0.4678850 subn=50
# subn 50 not norm 0.2478960 0.3578934 0.4678907

plot(pred.svm.test,test.cd41$y)
testlm <- lm(test.cd41$y~ pred.svm.test - 1)
abline(a = 0, b= 1)
abline(a = 0, b = testlm$coefficients[1])


var(test.cd41$y) #[1] 1.044508
ggplot(data.frame(cbind(pred.svm.test,test.cd41$y)), aes(x = pred.svm.test, y = test.cd41$y))+
  geom_point() +geom_abline(intercept = 0, slope = 1) + 
  labs(title = 'cd41 in test set vs predicted value - RF-SVM',x = 'Predicted value',y='True test value') 
cor(pred.svm.test,test.cd41$y) 
# 0.7935173 subn=50
# nsub 50 not norm 0.793538

R2.rpart <- 1 - sum((test.cd41$y-pred.svm.test)^2)/sum((test.cd41$y-mean(test.cd41$y))^2)
R2.rpart #0.6277297 subn=50
# nsub 50, not norm 0.6277131
##########################################
#keep(list=c('dat384','prmut0','rtmut0'),sure=T)
