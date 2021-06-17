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
total_data_normalize<- predict(preProcess(total_data[,1:3], method=c('YeoJohnson')), total_data)

library(psych)
#total_data.processed <- predict(preProcess(total_data_normalize[,! colnames(total_data_normalize) %in% c('arm')], 
 #                                          method=c('zv','nzv',"center", "scale"))
  #                              ,total_data_normalize)
total_data.processed <- predict(preProcess(total_data[,! colnames(total_data) %in% c('arm')], 
                                           method=c('zv','nzv',"center", "scale"))
                                ,total_data)
multi.hist(total_data.processed[,1:3])

############################################################################################
# index for train and test split
# Stratify data
seed <- 42
set.seed(seed)
library(caret)
strat_index <- createDataPartition(y = total_data$arm,list = FALSE,p = 0.75)

###########################################################################################
# data for cd42
data.cd42 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna1','lrna2','cd41')]
names(data.cd42)[3] <- 'y'
train.cd42 <- data.cd42[strat_index,]
test.cd42 <- data.cd42[-strat_index,]

############################################################################################
#creat 5 folds for cv
num_cv_folds = 5
set.seed(seed)
cv_split_cd42 <- vfold_cv(train.cd42, v=num_cv_folds, strata = y)
cv_data_cd42 <- cv_split_cd42 %>% 
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
##################################################################################
##################################################################################
#RANDOM FORREST WITH RANGER PACKAGE WITH cd42
library(ranger)

# Prepare for tuning the cross validation folds BY mtry, nodesize and sampsize
mtry_min <- 1
mtry_max <- floor(sqrt((ncol(train.cd42)-1)))
mtry <- seq(mtry_min,mtry_max, by =2)
nodesize <- seq(2,10,2)
numtree <- seq(300,800,100)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, numtree = numtree)

#CROSS THE ELEMENT OF HYPER GRID 
cv_tune_cd42 <- cv_data_cd42 %>%
  tidyr::crossing( mtry = hyper_grid$mtry,nodesize = hyper_grid$nodesize, numtree = hyper_grid$numtree) 


# Build RANDOM FOREST model for each fold & mtry/nodesize/sampsize combination 

cv_model_cd42 <- cv_tune_cd42 %>%
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
cv_model_cd42 <- cv_model_cd42 %>% 
  mutate(validate_predicted = map2(.x = model, .y = validate, ~ predict(.x, .y)$predictions),
         validate_actual = map(validate, ~ .x$y))  %>%
  # Calculate validate accuracy for each fold and mtry combination
  mutate(error = map2_dbl(.x = validate_actual, .y = validate_predicted, ~ mean((.x-.y)^2)))

cv_eval_cd42 <- cv_model_cd42 %>%
  group_by(numtree,mtry,nodesize) %>%
  summarise(mean_error = mean(error)) %>%
  ungroup(mtry,nodesize,numtree)


ggplot(data = cv_eval_cd42, aes(x=nodesize,y = mean_error)) + geom_line() + 
  facet_wrap(~numtree,labeller = label_both)+
  labs(title ="Mean error by nodesize", x = "nodesize", y = "Error") 


best_mtry = cv_eval_cd42$mtry[cv_eval_cd42$mean_error == min(cv_eval_cd42$mean_error)]
#best_mtry = floor(sqrt((ncol(train.cd42)-1)))
best_nodesize = cv_eval_cd42$nodesize[cv_eval_cd42$mean_error == min(cv_eval_cd42$mean_error)]
best_numtree = cv_eval_cd42$numtree[cv_eval_cd42$mean_error == min(cv_eval_cd42$mean_error)]

rffit <-  ranger(formula = y ~ ., 
                 data = train.cd42, 
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


####################################################################################
importance.rf <- sort(rffit$variable.importance,decreasing = T)

####################################################################################
# 95% CI
ntest <- length(test.cd42$y)
ntrain <- length(train.cd42$y)
#train error
cd42.pred.rf <- predict(rffit,train.cd42)$prediction
mean.mspe.cd42 <- mean((train.cd42$y - cd42.pred.rf)^2)
sigma.mspe.cd42 <- 1/ntest * var((train.cd42$y - cd42.pred.rf)^2) 
c(mean.mspe.cd42 - 1.96 * sqrt(sigma.mspe.cd42), mean.mspe.cd42, mean.mspe.cd42 + 1.96 * sqrt(sigma.mspe.cd42))
#[1] 0.1402103 0.1984074 0.2566045
# 0.1401784 0.1983728 0.2565672 NOT NORM
ggplot(data.frame(cbind(cd42.pred.rf,train.cd42$y)), aes(x = cd42.pred.rf, y = train.cd42$y))+
  geom_point() + ylim(-4,6) + xlim(-2,3) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'cd42 in train set vs predicted value - Random forest',x = 'Predicted value',y='True train value')

# test error
cd42.pred.rf <- predict(rffit,test.cd42)$prediction
mean.mspe.cd42 <- mean((test.cd42$y - cd42.pred.rf)^2)
sigma.mspe.cd42 <- 1/ntest * var((test.cd42$y - cd42.pred.rf)^2) 
c(mean.mspe.cd42 - 1.96 * sqrt(sigma.mspe.cd42), mean.mspe.cd42, mean.mspe.cd42 + 1.96 * sqrt(sigma.mspe.cd42))
# [1]0.4076962 0.5456170 0.6835379
# 0.4075271 0.5454629 0.6833988 NOT NORM
ggplot(data.frame(cbind(cd42.pred.rf,test.cd42$y)), aes(x = cd42.pred.rf, y = test.cd42$y))+
  geom_point() + ylim(-4,6)  +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'cd42 in test set vs predicted value - Random forest',x = 'Predicted value',y='True test value')
cor(cd42.pred.rf,test.cd42$y) #[1]  0.6598071 # NOT NORM 0.6598839
R2.rpart <- 1 - sum((test.cd42$y-cd42.pred.rf)^2)/sum((test.cd42$y-mean(test.cd42$y))^2)
R2.rpart #0.4110808 #NOT NORM 0.4112471



















##########################pick 20 variables to train###############################

top20_importance.cd42 <- names(importance.rf)[1:18]
data.cd42_importance <- data.cd42[, colnames(data.cd42) %in% c('y',top20_importance.cd42)]
train.cd42_importance <- data.cd42_importance[strat_index,]
test.cd42_importance <- data.cd42_importance[-strat_index,]

rffit <-  ranger(formula = y ~ ., 
                 data = train.cd42_importance, 
                 mtry = 5, #best_mtry, 
                 num.trees = 300, #best_numtree, 
                 min.node.size = 2, #best_nodesize,
                 seed = seed, 
                 oob.error = TRUE,
                 importance = 'impurity',
                 classification = FALSE,
                 replace = TRUE,
                 verbose = TRUE,
                 respect.unordered.factors = FALSE,
                 regularization.factor = 1)

cd42.pred.rf <- predict(rffit,test.cd42)$prediction

# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.cd42 <- mean((test.cd42$y - cd42.pred.rf)^2)
sigma.mspe.cd42 <- 1/ntest * var((test.cd42$y - cd42.pred.rf)^2) 
# 95% CI:
c(mean.mspe.cd42 - 1.96 * sqrt(sigma.mspe.cd42), mean.mspe.cd42, mean.mspe.cd42 + 1.96 * sqrt(sigma.mspe.cd42))
#[1] 32019.50 43092.85 54166.20 : top 30
# 31292.72 41560.46 51828.19 top 20
# 30987.73 41154.86 51321.98 top 19
# [1] 0.3856916 0.5192871 0.6528826 top 18
plot(cd42.pred.rf,test.cd42$y)
abline(a=0, b=1)
sum(cd42.pred.rf <= test.cd42$y) # 84
sum(cd42.pred.rf > test.cd42$y) # more overestimate 127

#keep(list=c('dat384','prmut0','rtmut0'),sure=T)
