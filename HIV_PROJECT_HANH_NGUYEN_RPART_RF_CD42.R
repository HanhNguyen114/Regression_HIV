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
# data for cd42
data.cd42 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna2','lrna1','cd41')]
names(data.cd42)[3] <- 'y'
#########################################################################################
# RPART
set.seed(42)
library(rpart)
parameter <- rpart.control(minsplit=400, xval=10, cp=0.01)
# [1] 0.3973572 0.4805780 0.5637989 vary=0.879025 minsplit=400, xval=10, cp=0.01 with nzv cor=0.6778548 R2=0.4506289

rpfit <- rpart(y ~., method="anova", control=parameter, data = data.cd42)
#printcp(rpfit)
plot(rpfit, uniform=T, branch=0.5, compress=T, margin=.1)
text(rpfit, all=T, use.n=F, col="blue",cex=0.8, fancy=T)
table(rpfit$where)

cl <- list()
cl[[1]] <- data.cd42[rpfit$where == 3,]
cl[[2]] <- data.cd42[rpfit$where == 4,]
cl[[3]] <- data.cd42[rpfit$where == 5,]
#cl[[4]] <- data.cd42[rpfit$where == 7,]
#[1] "frame"               "where"               "call"               
#[4] "terms"               "cptable"             "method"             
#[7] "parms"               "control"             "functions"          
#[10] "numresp"             "splits"              "csplit"             
#[13] "variable.importance" "y"                   "ordered" 
############################################################################################
# index for train and test split
# Stratify data
seed <- 42
set.seed(seed)
library(caret)
strat_index <- list()
for (i in 1:length(cl)) {
strat_index[[i]] <- createDataPartition(y = cl[[i]]$arm,list = FALSE,p = 0.75)
}
###########################################################################################
# split train test slit
train.cd42 <- list()
test.cd42 <- list()
for (i in 1:length(cl)) {
  train.cd42[[i]] <- cl[[i]][strat_index[[i]],]
  test.cd42[[i]] <- cl[[i]][-strat_index[[i]],]
}

table(train.cd42[[1]]$arm)
table(train.cd42[[2]]$arm)
table(train.cd42[[3]]$arm)
#table(train.cd42[[4]]$arm)
#########################################################################################
library(ranger)
cv_split_cd42 <- list()
cv_data_cd42 <- list()
cv_tune_cd42 <- list()
cv_model_cd42 <- list()
cv_eval_cd42 <- list()
cd42.pred.rf.train <- list()
cd42.pred.rf.test <- list()
rffit <- list()
num_cv_folds = 3
mtry <- 1
nodesize <- seq(2,10,2)
numtree <- seq(300,800,100)
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, numtree = numtree)
set.seed(seed)

for (i in 1:length(cl)) {
  ######################creat 5 folds for cv
  cv_split_cd42[[i]] <- vfold_cv(train.cd42[[i]], v=num_cv_folds, strata = y)
  cv_data_cd42[[i]] <- cv_split_cd42[[i]] %>% 
    mutate(
      # Extract the train dataframe for each split
      train = map(splits, ~training(.x)), 
      # Extract the validate dataframe for each split
      validate = map(splits, ~testing(.x))
    )  %>%
    mutate(
      #N-train of train set
      n_train = map_dbl(train, nrow),
      #N-validate of validate set
      n_validate = map_dbl(validate, nrow)
    )
  ################################CROSS THE ELEMENT OF HYPER GRID 
  cv_tune_cd42[[i]] <- cv_data_cd42[[i]] %>%
    tidyr::crossing( mtry = hyper_grid$mtry,nodesize = hyper_grid$nodesize, numtree = hyper_grid$numtree) 
  ################################ Build RANDOM FOREST model for each fold & mtry/nodesize/sampsize combination 
  cv_model_cd42[[i]] <- cv_tune_cd42[[i]] %>%
    mutate(.l = as.list(data.frame(t(cbind(mtry,nodesize,numtree))))) %>%
    mutate(model = map2(.x = train, .y = .l,~ ranger(formula = y~., 
                                                     data = .x, 
                                                     mtry = .y[1], 
                                                     num.trees = .y[3], 
                                                     min.node.size = .y[2],
                                                     seed = seed, 
                                                     oob.error = FALSE,
                                                     importance = 'impurity',
                                                     classification = FALSE,
                                                     replace = TRUE,
                                                     verbose = TRUE )))
  cv_model_cd42[[i]] <- cv_model_cd42[[i]] %>% 
    mutate(validate_predicted = map2(.x = model, .y = validate, ~ predict(.x, .y)$predictions),
           validate_actual = map(validate, ~ .x$y))  %>%
    ##################### Calculate validate accuracy for each fold and mtry combination
    mutate(error = map2_dbl(.x = validate_actual, .y = validate_predicted, ~ mean((.x-.y)^2)))
  cv_eval_cd42[[i]] <- cv_model_cd42[[i]] %>%
    group_by(numtree,mtry,nodesize) %>%
    summarise(mean_error = mean(error)) %>%
    ungroup(mtry,nodesize,numtree)
  ##################### Best tuning parameters
  best_mtry = cv_eval_cd42[[i]]$mtry[cv_eval_cd42[[i]]$mean_error == min(cv_eval_cd42[[i]]$mean_error)]
  best_nodesize = cv_eval_cd42[[i]]$nodesize[cv_eval_cd42[[i]]$mean_error == min(cv_eval_cd42[[i]]$mean_error)]
  best_numtree = cv_eval_cd42[[i]]$numtree[cv_eval_cd42[[i]]$mean_error == min(cv_eval_cd42[[i]]$mean_error)]
  ##################### Fitting RF
  rffit[[i]] <-  ranger(formula = y ~ ., 
                   data = train.cd42[[i]], 
                   mtry = best_mtry, 
                   num.trees = best_numtree, 
                   min.node.size = best_nodesize,
                   seed = seed, 
                   oob.error = F,
                   importance = 'impurity',
                   classification = FALSE,
                   replace = TRUE,
                   verbose = FALSE,
                   respect.unordered.factors = FALSE,
                   regularization.factor = 1,
                   splitrule = "variance")
  #################### Predict
  cd42.pred.rf.train[[i]] <- predict(rffit[[i]],train.cd42[[i]])$prediction
  cd42.pred.rf.test[[i]] <- predict(rffit[[i]],test.cd42[[i]])$prediction
}
#############################################################
# train error
ntrain <- 0; ntest <- 0; pred.rf.train <- NULL; train.cd42.y <- NULL
pred.rf.test <- NULL; test.cd42.y <- NULL
for (i in 1:length(cl)) {
  ntrain <- ntrain + nrow(train.cd42[[i]])
  ntest <- ntest + nrow(test.cd42[[i]])
  pred.rf.train <- c(pred.rf.train,cd42.pred.rf.train[[i]])
  train.cd42.y <- c(train.cd42.y,train.cd42[[i]]$y)
  pred.rf.test <- c(pred.rf.test,cd42.pred.rf.test[[i]])
  test.cd42.y <- c(test.cd42.y,test.cd42[[i]]$y)
}
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.cd42.rf <- mean((train.cd42.y - pred.rf.train)^2)
sigma.mspe.cd42.rf <- 1/ntrain * var((train.cd42.y - pred.rf.train)^2) 
# 95% CI:
c(mean.mspe.cd42.rf - 1.96 * sqrt(sigma.mspe.cd42.rf), mean.mspe.cd42.rf, mean.mspe.cd42.rf + 1.96 * sqrt(sigma.mspe.cd42.rf))

plot(pred.rf.train,train.cd42.y)
trainlm <- lm(train.cd42.y~ pred.rf.train - 1)
abline(a= 0, b=1)
abline(a = 0, b = trainlm$coefficients[1])
#############################################################
# test error
mean.mspe.cd42.rf <- mean((test.cd42.y - pred.rf.test)^2)
sigma.mspe.cd42.rf <- 1/ntest * var((test.cd42.y - pred.rf.test)^2) 
c(mean.mspe.cd42.rf - 1.96 * sqrt(sigma.mspe.cd42.rf), mean.mspe.cd42.rf, mean.mspe.cd42.rf + 1.96 * sqrt(sigma.mspe.cd42.rf))
plot(pred.rf.test,test.cd42.y)
testlm <- lm(test.cd42.y ~ pred.rf.test -1)
abline(a = 0, b= testlm$coefficients[1])
abline(a = 0, b = 1)
sum(pred.rf.test <= test.cd42.y) # 68
sum(pred.rf.test > test.cd42.y) # more overestimate 143

R2.rpart <- 1 - sum((test.cd42.y-pred.rf.test)^2)/sum((test.cd42.y-mean(test.cd42.y))^2)
R2.rpart
#0.5395638

var(test.cd42.y)
ggplot(data.frame(cbind(pred.rf.test,test.cd42.y)), aes(x = pred.rf.test, y = test.cd42.y))+
  geom_point() + ylim(-3,5) + xlim(-3,3) +geom_abline(intercept = 0, slope = testlm$coefficients[1])
cor(pred.rf.test,test.cd42.y) #0.6778548

###############################################################################
###############################################################################
###############################################################################

#only rf, no depart
total.train <- Reduce(rbind,train.cd42)
total.test <- Reduce(rbind,test.cd42)

######################creat 5 folds for cv
cv_split_cd42.tot <- vfold_cv(total.train, v=num_cv_folds, strata = y)
cv_data_cd42.tot <- cv_split_cd42.tot %>% 
  mutate(
    # Extract the train dataframe for each split
    train = map(splits, ~training(.x)), 
    # Extract the validate dataframe for each split
    validate = map(splits, ~testing(.x))
  )  %>%
  mutate(
    #N-train of train set
    n_train = map_dbl(train, nrow),
    #N-validate of validate set
    n_validate = map_dbl(validate, nrow)
  )
################################CROSS THE ELEMENT OF HYPER GRID 
cv_tune_cd42.tot <- cv_data_cd42.tot %>%
  tidyr::crossing( mtry = hyper_grid$mtry,nodesize = hyper_grid$nodesize, numtree = hyper_grid$numtree) 
################################ Build RANDOM FOREST model for each fold & mtry/nodesize/sampsize combination 
cv_model_cd42.tot <- cv_tune_cd42.tot %>%
  mutate(.l = as.list(data.frame(t(cbind(mtry,nodesize,numtree))))) %>%
  mutate(model = map2(.x = train, .y = .l,~ ranger(formula = y~., 
                                                   data = .x, 
                                                   mtry = .y[1], 
                                                   num.trees = .y[3], 
                                                   min.node.size = .y[2],
                                                   seed = seed, 
                                                   oob.error = FALSE,
                                                   importance = 'impurity',
                                                   classification = FALSE,
                                                   replace = TRUE,
                                                   verbose = TRUE )))
cv_model_cd42.tot <- cv_model_cd42.tot %>% 
  mutate(validate_predicted = map2(.x = model, .y = validate, ~ predict(.x, .y)$predictions),
         validate_actual = map(validate, ~ .x$y))  %>%
  ##################### Calculate validate accuracy for each fold and mtry combination
  mutate(error = map2_dbl(.x = validate_actual, .y = validate_predicted, ~ mean((.x-.y)^2)))
cv_eval_cd42.tot <- cv_model_cd42.tot %>%
  group_by(numtree,mtry,nodesize) %>%
  summarise(mean_error = mean(error)) %>%
  ungroup(mtry,nodesize,numtree)
##################### Best tuning parameters
best_mtry = cv_eval_cd42.tot$mtry[cv_eval_cd42.tot$mean_error == min(cv_eval_cd42.tot$mean_error)]
best_nodesize = cv_eval_cd42.tot$nodesize[cv_eval_cd42.tot$mean_error == min(cv_eval_cd42.tot$mean_error)]
best_numtree = cv_eval_cd42.tot$numtree[cv_eval_cd42.tot$mean_error == min(cv_eval_cd42.tot$mean_error)]
##################### Fitting RF
rffit.tot <-  ranger(formula = y ~ ., 
                      data = total.train, 
                      mtry = best_mtry, 
                      num.trees = best_numtree, 
                      min.node.size = best_nodesize,
                      seed = seed, 
                      oob.error = F,
                      importance = 'impurity',
                      classification = FALSE,
                      replace = TRUE,
                      verbose = FALSE,
                      respect.unordered.factors = FALSE,
                      regularization.factor = 1,
                      splitrule = "variance")
#################### Predict
pred.rf.totr <- predict(rffit.tot,total.train)$prediction
pred.rf.tote <- predict(rffit.tot,total.test)$prediction


### train error
mean.rf <- mean((total.train$y - pred.rf.totr)^2)
sigma.rf <- 1/ntrain * var((total.train$y - pred.rf.totr)^2) 
c(mean.rf - 1.96 * sqrt(sigma.rf), mean.rf, mean.rf + 1.96 * sqrt(sigma.rf))
plot(pred.rf.totr,total.train$y)
trainlm <- lm(total.train$y~ pred.rf.totr - 1)
abline(a= 0, b=1)
abline(a = 0, b = trainlm$coefficients[1])

### test error
mean.rf <- mean((total.test$y - pred.rf.tote)^2)
sigma.rf <- 1/ntest * var((total.test$y - pred.rf.tote)^2) 
c(mean.rf - 1.96 * sqrt(sigma.rf), mean.rf, mean.rf + 1.96 * sqrt(sigma.rf))
plot(pred.rf.tote,total.test$y)
testlm <- lm(total.test$y~ pred.rf.tote - 1)
abline(a= 0, b=1)
abline(a = 0, b = testlm$coefficients[1])
var(total.test$y) #[1] 0.879025
# [1] 0.6383162 0.7694722 0.9006281 test error

ggplot(data.frame(cbind(pred.rf.tote,y = total.test$y)), aes(x = pred.rf.tote, y = y))+
  geom_point() + ylim(-3,5) + xlim(-1,1) +geom_abline(intercept = 0, slope = testlm$coefficients[1])

cor(pred.rf.tote,total.test$y) #0.5813635

R2.0rpart <- 1 - sum((total.test$y-pred.rf.tote)^2)/sum((total.test$y-mean(total.test$y))^2)
R2.0rpart #0.1203806
