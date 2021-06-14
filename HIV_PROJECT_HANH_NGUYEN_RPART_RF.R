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
total_data.processed <- predict(preProcess(total_data_normalize[,! colnames(total_data_normalize) %in% c('arm')], 
                                           method=c('zv','nzv',"center", "scale"))
                          ,total_data_normalize)
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
# RPART
set.seed(42)
library(rpart)
parameter <- rpart.control(minsplit=300, xval=10, cp=0.01)
# *****[1] 0.6777456 0.9129387 1.1481318 vary=0.9049796 minsplit=300, xval=10, cp=0.01 with nzv (test err) cor=0.1209692
# [1] 0.6753118 0.9111353 1.1469587 minsplit=300, xval=10, cp=0.01 without nzv
# [1] 0.6849698 0.9743451 1.2637204 very=1.00568 minsplit=300, xval=10, cp=0.001 with nzv cor=0.1637727
# [1] 0.7754332 0.9350653 1.0946975 minsplit=300, xval=10, cp=0.01 with norm lrna
# [1] 0.7765803 0.9488311 1.1210820 minsplit=400, xval=10, cp=0.001 with norm lrna
rpfit <- rpart(y ~., method="anova", control=parameter, data = data.lrna1)
#printcp(rpfit)
plot(rpfit, uniform=T, branch=0.5, compress=T, margin=.1)
text(rpfit, all=T, use.n=F, col="blue",cex=0.8, fancy=T)
table(rpfit$where)

cl <- list()
cl[[1]] <- data.lrna1[rpfit$where == 2,]
cl[[2]] <- data.lrna1[rpfit$where == 4,]
cl[[3]] <- data.lrna1[rpfit$where == 5,]
#cl[[4]] <- data.lrna1[rpfit$where == 7,]
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
train.lrna1 <- list()
test.lrna1 <- list()
for (i in 1:length(cl)) {
  train.lrna1[[i]] <- cl[[i]][strat_index[[i]],]
  test.lrna1[[i]] <- cl[[i]][-strat_index[[i]],]
}

table(train.lrna1[[1]]$arm)
table(train.lrna1[[2]]$arm)
table(train.lrna1[[3]]$arm)
#table(train.lrna1[[4]]$arm)
#########################################################################################
library(ranger)
cv_split_lrna1 <- list()
cv_data_lrna1 <- list()
cv_tune_lrna1 <- list()
cv_model_lrna1 <- list()
cv_eval_lrna1 <- list()
lrna1.pred.rf.train <- list()
lrna1.pred.rf.test <- list()
rffit <- list()
num_cv_folds = 3
mtry <- 1
nodesize <- seq(2,10,2)
numtree <- seq(300,800,100)
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, numtree = numtree)
set.seed(seed)

for (i in 1:length(cl)) {
  ######################creat 5 folds for cv
  cv_split_lrna1[[i]] <- vfold_cv(train.lrna1[[i]], v=num_cv_folds, strata = y)
  cv_data_lrna1[[i]] <- cv_split_lrna1[[i]] %>% 
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
  cv_tune_lrna1[[i]] <- cv_data_lrna1[[i]] %>%
    tidyr::crossing( mtry = hyper_grid$mtry,nodesize = hyper_grid$nodesize, numtree = hyper_grid$numtree) 
  ################################ Build RANDOM FOREST model for each fold & mtry/nodesize/sampsize combination 
  cv_model_lrna1[[i]] <- cv_tune_lrna1[[i]] %>%
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
  cv_model_lrna1[[i]] <- cv_model_lrna1[[i]] %>% 
    mutate(validate_predicted = map2(.x = model, .y = validate, ~ predict(.x, .y)$predictions),
           validate_actual = map(validate, ~ .x$y))  %>%
    ##################### Calculate validate accuracy for each fold and mtry combination
    mutate(error = map2_dbl(.x = validate_actual, .y = validate_predicted, ~ mean((.x-.y)^2)))
  cv_eval_lrna1[[i]] <- cv_model_lrna1[[i]] %>%
    group_by(numtree,mtry,nodesize) %>%
    summarise(mean_error = mean(error)) %>%
    ungroup(mtry,nodesize,numtree)
  ##################### Best tuning parameters
  best_mtry = cv_eval_lrna1[[i]]$mtry[cv_eval_lrna1[[i]]$mean_error == min(cv_eval_lrna1[[i]]$mean_error)]
  best_nodesize = cv_eval_lrna1[[i]]$nodesize[cv_eval_lrna1[[i]]$mean_error == min(cv_eval_lrna1[[i]]$mean_error)]
  best_numtree = cv_eval_lrna1[[i]]$numtree[cv_eval_lrna1[[i]]$mean_error == min(cv_eval_lrna1[[i]]$mean_error)]
  ##################### Fitting RF
  rffit[[i]] <-  ranger(formula = y ~ ., 
                   data = train.lrna1[[i]], 
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
  lrna1.pred.rf.train[[i]] <- predict(rffit[[i]],train.lrna1[[i]])$prediction
  lrna1.pred.rf.test[[i]] <- predict(rffit[[i]],test.lrna1[[i]])$prediction
}
#############################################################
# train error
ntrain <- 0; ntest <- 0; pred.rf.train <- NULL; train.lrna1.y <- NULL
pred.rf.test <- NULL; test.lrna1.y <- NULL
for (i in 1:length(cl)) {
  ntrain <- ntrain + nrow(train.lrna1[[i]])
  ntest <- ntest + nrow(test.lrna1[[i]])
  pred.rf.train <- c(pred.rf.train,lrna1.pred.rf.train[[i]])
  train.lrna1.y <- c(train.lrna1.y,train.lrna1[[i]]$y)
  pred.rf.test <- c(pred.rf.test,lrna1.pred.rf.test[[i]])
  test.lrna1.y <- c(test.lrna1.y,test.lrna1[[i]]$y)
}
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna1.rf <- mean((train.lrna1.y - pred.rf.train)^2)
sigma.mspe.lrna1.rf <- 1/ntrain * var((train.lrna1.y - pred.rf.train)^2) 
# 95% CI:
c(mean.mspe.lrna1.rf - 1.96 * sqrt(sigma.mspe.lrna1.rf), mean.mspe.lrna1.rf, mean.mspe.lrna1.rf + 1.96 * sqrt(sigma.mspe.lrna1.rf))
plot(pred.rf.train,train.lrna1.y)
trainlm <- lm(train.lrna1.y~ pred.rf.train - 1)
abline(a= 0, b=1)
abline(a = 0, b = trainlm$coefficients[1])
#############################################################
# test error
mean.mspe.lrna1.rf <- mean((test.lrna1.y - pred.rf.test)^2)
sigma.mspe.lrna1.rf <- 1/ntest * var((test.lrna1.y - pred.rf.test)^2) 
c(mean.mspe.lrna1.rf - 1.96 * sqrt(sigma.mspe.lrna1.rf), mean.mspe.lrna1.rf, mean.mspe.lrna1.rf + 1.96 * sqrt(sigma.mspe.lrna1.rf))
plot(pred.rf.test,test.lrna1.y)
testlm <- lm(test.lrna1.y ~ pred.rf.test -1)
abline(a = 0, b= testlm$coefficients[1])
abline(a = 0, b = 1)
sum(pred.rf.test <= test.lrna1.y) # 68
sum(pred.rf.test > test.lrna1.y) # more overestimate 143

R2.rpart <- 1 - sum((test.lrna1.y-pred.rf.test)^2)/sum((test.lrna1.y-mean(test.lrna1.y))^2)
R2.rpart #-0.01362161 or 0.004522969 with norm lrna

var(test.lrna1.y)
ggplot(data.frame(cbind(pred.rf.test,test.lrna1.y)), aes(x = pred.rf.test, y = test.lrna1.y))+
  geom_point() + ylim(-1,5) + xlim(-1,2) +geom_abline(intercept = 0, slope = testlm$coefficients[1])
cor(pred.rf.test,test.lrna1.y) #0.1209692 or 0.1429455 with norm lrna

###############################################################################
###############################################################################
###############################################################################

#only rf, no depart
total.train <- Reduce(rbind,train.lrna1)
total.test <- Reduce(rbind,test.lrna1)

######################creat 5 folds for cv
cv_split_lrna1.tot <- vfold_cv(total.train, v=num_cv_folds, strata = y)
cv_data_lrna1.tot <- cv_split_lrna1.tot %>% 
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
cv_tune_lrna1.tot <- cv_data_lrna1.tot %>%
  tidyr::crossing( mtry = hyper_grid$mtry,nodesize = hyper_grid$nodesize, numtree = hyper_grid$numtree) 
################################ Build RANDOM FOREST model for each fold & mtry/nodesize/sampsize combination 
cv_model_lrna1.tot <- cv_tune_lrna1.tot %>%
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
cv_model_lrna1.tot <- cv_model_lrna1.tot %>% 
  mutate(validate_predicted = map2(.x = model, .y = validate, ~ predict(.x, .y)$predictions),
         validate_actual = map(validate, ~ .x$y))  %>%
  ##################### Calculate validate accuracy for each fold and mtry combination
  mutate(error = map2_dbl(.x = validate_actual, .y = validate_predicted, ~ mean((.x-.y)^2)))
cv_eval_lrna1.tot <- cv_model_lrna1.tot %>%
  group_by(numtree,mtry,nodesize) %>%
  summarise(mean_error = mean(error)) %>%
  ungroup(mtry,nodesize,numtree)
##################### Best tuning parameters
best_mtry = cv_eval_lrna1.tot$mtry[cv_eval_lrna1.tot$mean_error == min(cv_eval_lrna1.tot$mean_error)]
best_nodesize = cv_eval_lrna1.tot$nodesize[cv_eval_lrna1.tot$mean_error == min(cv_eval_lrna1.tot$mean_error)]
best_numtree = cv_eval_lrna1.tot$numtree[cv_eval_lrna1.tot$mean_error == min(cv_eval_lrna1.tot$mean_error)]
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
# [1] 0.6802432 0.9249037 1.1695643
# [1] 0.7982631 0.9554271 1.1125911 with norm lrna
plot(pred.rf.tote,total.test$y)
testlm <- lm(total.test$y~ pred.rf.tote - 1)
abline(a= 0, b=1)
abline(a = 0, b = testlm$coefficients[1])
var(total.test$y) #[1] 0.9049796


ggplot(data.frame(cbind(pred.rf.tote,y = total.test$y)), aes(x = pred.rf.tote, y = y))+
  geom_point() + ylim(-1,5) + xlim(-1,2) +geom_abline(intercept = 0, slope = testlm$coefficients[1])
cor(pred.rf.tote,total.test$y) #-0.005140603 or -0.03412849 with norm lrna

R2.0rpart <- 1 - sum((total.test$y-pred.rf.tote)^2)/sum((total.test$y-mean(total.test$y))^2)
R2.0rpart #-0.02690613; -0.01715429 with norm lrna
