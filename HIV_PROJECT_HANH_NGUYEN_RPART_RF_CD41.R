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
# data for cd41
data.cd41 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna2','lrna1','cd42')]
names(data.cd41)[3] <- 'y'
#########################################################################################
# RPART
set.seed(42)
library(rpart)
parameter <- rpart.control(minsplit=400, xval=10, cp=0.01)
# [1] 0.3421587 0.5779802 0.8138017 vary=1.261442 minsplit=300, xval=10, cp=0.01 with nzv cor=0.7458803

rpfit <- rpart(y ~., method="anova", control=parameter, data = data.cd41)
#printcp(rpfit)
plot(rpfit, uniform=T, branch=0.5, compress=T, margin=.1)
text(rpfit, all=T, use.n=F, col="blue",cex=0.8, fancy=T)
table(rpfit$where)

cl <- list()
cl[[1]] <- data.cd41[rpfit$where == 3,]
cl[[2]] <- data.cd41[rpfit$where == 4,]
cl[[3]] <- data.cd41[rpfit$where == 5,]
#cl[[4]] <- data.cd41[rpfit$where == 7,]
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
train.cd41 <- list()
test.cd41 <- list()
for (i in 1:length(cl)) {
  train.cd41[[i]] <- cl[[i]][strat_index[[i]],]
  test.cd41[[i]] <- cl[[i]][-strat_index[[i]],]
}

table(train.cd41[[1]]$arm)
table(train.cd41[[2]]$arm)
table(train.cd41[[3]]$arm)
#table(train.cd41[[4]]$arm)
#########################################################################################
library(ranger)
cv_split_cd41 <- list()
cv_data_cd41 <- list()
cv_tune_cd41 <- list()
cv_model_cd41 <- list()
cv_eval_cd41 <- list()
cd41.pred.rf.train <- list()
cd41.pred.rf.test <- list()
rffit <- list()
num_cv_folds = 3
mtry <- 1
nodesize <- seq(2,10,2)
numtree <- seq(300,800,100)
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, numtree = numtree)
set.seed(seed)

for (i in 1:length(cl)) {
  ######################creat 5 folds for cv
  cv_split_cd41[[i]] <- vfold_cv(train.cd41[[i]], v=num_cv_folds, strata = y)
  cv_data_cd41[[i]] <- cv_split_cd41[[i]] %>% 
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
  cv_tune_cd41[[i]] <- cv_data_cd41[[i]] %>%
    tidyr::crossing( mtry = hyper_grid$mtry,nodesize = hyper_grid$nodesize, numtree = hyper_grid$numtree) 
  ################################ Build RANDOM FOREST model for each fold & mtry/nodesize/sampsize combination 
  cv_model_cd41[[i]] <- cv_tune_cd41[[i]] %>%
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
  cv_model_cd41[[i]] <- cv_model_cd41[[i]] %>% 
    mutate(validate_predicted = map2(.x = model, .y = validate, ~ predict(.x, .y)$predictions),
           validate_actual = map(validate, ~ .x$y))  %>%
    ##################### Calculate validate accuracy for each fold and mtry combination
    mutate(error = map2_dbl(.x = validate_actual, .y = validate_predicted, ~ mean((.x-.y)^2)))
  cv_eval_cd41[[i]] <- cv_model_cd41[[i]] %>%
    group_by(numtree,mtry,nodesize) %>%
    summarise(mean_error = mean(error)) %>%
    ungroup(mtry,nodesize,numtree)
  ##################### Best tuning parameters
  best_mtry = cv_eval_cd41[[i]]$mtry[cv_eval_cd41[[i]]$mean_error == min(cv_eval_cd41[[i]]$mean_error)]
  best_nodesize = cv_eval_cd41[[i]]$nodesize[cv_eval_cd41[[i]]$mean_error == min(cv_eval_cd41[[i]]$mean_error)]
  best_numtree = cv_eval_cd41[[i]]$numtree[cv_eval_cd41[[i]]$mean_error == min(cv_eval_cd41[[i]]$mean_error)]
  ##################### Fitting RF
  rffit[[i]] <-  ranger(formula = y ~ ., 
                   data = train.cd41[[i]], 
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
  cd41.pred.rf.train[[i]] <- predict(rffit[[i]],train.cd41[[i]])$prediction
  cd41.pred.rf.test[[i]] <- predict(rffit[[i]],test.cd41[[i]])$prediction
}
#############################################################
# train error
ntrain <- 0; ntest <- 0; pred.rf.train <- NULL; train.cd41.y <- NULL
pred.rf.test <- NULL; test.cd41.y <- NULL
for (i in 1:length(cl)) {
  ntrain <- ntrain + nrow(train.cd41[[i]])
  ntest <- ntest + nrow(test.cd41[[i]])
  pred.rf.train <- c(pred.rf.train,cd41.pred.rf.train[[i]])
  train.cd41.y <- c(train.cd41.y,train.cd41[[i]]$y)
  pred.rf.test <- c(pred.rf.test,cd41.pred.rf.test[[i]])
  test.cd41.y <- c(test.cd41.y,test.cd41[[i]]$y)
}
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.cd41.rf <- mean((train.cd41.y - pred.rf.train)^2)
sigma.mspe.cd41.rf <- 1/ntrain * var((train.cd41.y - pred.rf.train)^2) 
# 95% CI:
c(mean.mspe.cd41.rf - 1.96 * sqrt(sigma.mspe.cd41.rf), mean.mspe.cd41.rf, mean.mspe.cd41.rf + 1.96 * sqrt(sigma.mspe.cd41.rf))

plot(pred.rf.train,train.cd41.y)
trainlm <- lm(train.cd41.y~ pred.rf.train - 1)
abline(a= 0, b=1)
abline(a = 0, b = trainlm$coefficients[1])
#############################################################
# test error
mean.mspe.cd41.rf <- mean((test.cd41.y - pred.rf.test)^2)
sigma.mspe.cd41.rf <- 1/ntest * var((test.cd41.y - pred.rf.test)^2) 
c(mean.mspe.cd41.rf - 1.96 * sqrt(sigma.mspe.cd41.rf), mean.mspe.cd41.rf, mean.mspe.cd41.rf + 1.96 * sqrt(sigma.mspe.cd41.rf))
plot(pred.rf.test,test.cd41.y)
testlm <- lm(test.cd41.y ~ pred.rf.test -1)
abline(a = 0, b= testlm$coefficients[1])
abline(a = 0, b = 1)
sum(pred.rf.test <= test.cd41.y) # 68
sum(pred.rf.test > test.cd41.y) # more overestimate 143

R2.rpart <- 1 - sum((test.cd41.y-pred.rf.test)^2)/sum((test.cd41.y-mean(test.cd41.y))^2)
R2.rpart
#0.5395638

var(test.cd41.y)
ggplot(data.frame(cbind(pred.rf.test,test.cd41.y)), aes(x = pred.rf.test, y = test.cd41.y))+
  geom_point() + ylim(-1,5) + xlim(-1,2) +geom_abline(intercept = 0, slope = testlm$coefficients[1])
cor(pred.rf.test,test.cd41.y) #0.7458803

###############################################################################
###############################################################################
###############################################################################

#only rf, no depart
total.train <- Reduce(rbind,train.cd41)
total.test <- Reduce(rbind,test.cd41)

######################creat 5 folds for cv
cv_split_cd41.tot <- vfold_cv(total.train, v=num_cv_folds, strata = y)
cv_data_cd41.tot <- cv_split_cd41.tot %>% 
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
cv_tune_cd41.tot <- cv_data_cd41.tot %>%
  tidyr::crossing( mtry = hyper_grid$mtry,nodesize = hyper_grid$nodesize, numtree = hyper_grid$numtree) 
################################ Build RANDOM FOREST model for each fold & mtry/nodesize/sampsize combination 
cv_model_cd41.tot <- cv_tune_cd41.tot %>%
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
cv_model_cd41.tot <- cv_model_cd41.tot %>% 
  mutate(validate_predicted = map2(.x = model, .y = validate, ~ predict(.x, .y)$predictions),
         validate_actual = map(validate, ~ .x$y))  %>%
  ##################### Calculate validate accuracy for each fold and mtry combination
  mutate(error = map2_dbl(.x = validate_actual, .y = validate_predicted, ~ mean((.x-.y)^2)))
cv_eval_cd41.tot <- cv_model_cd41.tot %>%
  group_by(numtree,mtry,nodesize) %>%
  summarise(mean_error = mean(error)) %>%
  ungroup(mtry,nodesize,numtree)
##################### Best tuning parameters
best_mtry = cv_eval_cd41.tot$mtry[cv_eval_cd41.tot$mean_error == min(cv_eval_cd41.tot$mean_error)]
best_nodesize = cv_eval_cd41.tot$nodesize[cv_eval_cd41.tot$mean_error == min(cv_eval_cd41.tot$mean_error)]
best_numtree = cv_eval_cd41.tot$numtree[cv_eval_cd41.tot$mean_error == min(cv_eval_cd41.tot$mean_error)]
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
var(total.test$y) #[1] 1.261442
# [1] 0.7494615 1.0961733 1.4428852 test error

ggplot(data.frame(cbind(pred.rf.tote,y = total.test$y)), aes(x = pred.rf.tote, y = y))+
  geom_point() + ylim(-3,5) + xlim(-1,1) +geom_abline(intercept = 0, slope = testlm$coefficients[1])

cor(pred.rf.tote,total.test$y) #0.633273

R2.0rpart <- 1 - sum((total.test$y-pred.rf.tote)^2)/sum((total.test$y-mean(total.test$y))^2)
R2.0rpart #0.1267559
