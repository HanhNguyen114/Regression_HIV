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
total_data.processed <- predict(preProcess(total_data_normalize[,! colnames(total_data_normalize) %in% c('arm')], 
                                           method=c('zv','nzv',"center", "scale"))
                                ,total_data_normalize)
#total_data.processed <- predict(preProcess(total_data[,! colnames(total_data) %in% c('arm')], 
 #                                          method=c('zv','nzv',"center", "scale"))
  #                              ,total_data)
multi.hist(total_data.processed[,1:3])

############################################################################################
# index for train and test split
# Stratify data
seed <- 42
set.seed(seed)
library(caret)
strat_index <- createDataPartition(y = total_data$arm,list = FALSE,p = 0.75)

###########################################################################################
# data for cd41
data.cd41 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna1','lrna2','cd42')]
names(data.cd41)[3] <- 'y'
train.cd41 <- data.cd41[strat_index,]
test.cd41 <- data.cd41[-strat_index,]

############################################################################################
#creat 5 folds for cv
num_cv_folds = 5
set.seed(seed)
cv_split_cd41 <- vfold_cv(train.cd41, v=num_cv_folds, strata = y)
cv_data_cd41 <- cv_split_cd41 %>% 
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
#RANDOM FORREST WITH RANGER PACKAGE WITH cd41
library(ranger)

# Prepare for tuning the cross validation folds BY mtry, nodesize and sampsize
mtry_min <- 1
mtry_max <- floor(sqrt((ncol(train.cd41)-1)))
mtry <- seq(mtry_min,mtry_max,by = 2)
nodesize <- seq(2,10,2)
numtree <- seq(300,800,100)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, numtree = numtree)

#CROSS THE ELEMENT OF HYPER GRID 
cv_tune_cd41 <- cv_data_cd41 %>%
  tidyr::crossing( mtry = hyper_grid$mtry,nodesize = hyper_grid$nodesize, numtree = hyper_grid$numtree) 


# Build RANDOM FOREST model for each fold & mtry/nodesize/sampsize combination 

cv_model_cd41 <- cv_tune_cd41 %>%
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
cv_model_cd41 <- cv_model_cd41 %>% 
  mutate(validate_predicted = map2(.x = model, .y = validate, ~ predict(.x, .y)$predictions),
         validate_actual = map(validate, ~ .x$y))  %>%
  # Calculate validate accuracy for each fold and mtry combination
  mutate(error = map2_dbl(.x = validate_actual, .y = validate_predicted, ~ mean((.x-.y)^2)))

cv_eval_cd41 <- cv_model_cd41 %>%
  group_by(numtree,mtry,nodesize) %>%
  summarise(mean_error = mean(error)) %>%
  ungroup(mtry,nodesize,numtree)


ggplot(data = cv_eval_cd41, aes(x=nodesize,y = mean_error)) + geom_line() + 
  facet_wrap(~numtree,labeller = label_both)+
  labs(title ="Mean error by nodesize", x = "nodesize", y = "Error") 


best_mtry = cv_eval_cd41$mtry[cv_eval_cd41$mean_error == min(cv_eval_cd41$mean_error)]
#best_mtry = floor(sqrt((ncol(train.cd41)-1)))
best_nodesize = cv_eval_cd41$nodesize[cv_eval_cd41$mean_error == min(cv_eval_cd41$mean_error)]
best_numtree = cv_eval_cd41$numtree[cv_eval_cd41$mean_error == min(cv_eval_cd41$mean_error)]

rffit <-  ranger(formula = y ~ ., 
                 data = train.cd41, 
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
ntest <- length(test.cd41$y)
ntrain <- length(train.cd41$y)
#train error

cd41.pred.rf <- predict(rffit,train.cd41)$prediction
mean.mspe.cd41 <- mean((train.cd41$y - cd41.pred.rf)^2)
sigma.mspe.cd41 <- 1/ntest * var((train.cd41$y - cd41.pred.rf)^2) 
c(mean.mspe.cd41 - 1.96 * sqrt(sigma.mspe.cd41), mean.mspe.cd41, mean.mspe.cd41 + 1.96 * sqrt(sigma.mspe.cd41))
ggplot(data.frame(cbind(cd41.pred.rf,train.cd41$y)), aes(x = cd41.pred.rf, y = train.cd41$y))+
  geom_point() + ylim(-4,6) + xlim(-1,1) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'cd41 in train set vs predicted value - Random forest',x = 'Predicted value',y='True train value')
#[1] 0.05454130 0.07723582 0.09993034
# [1] 0.05459292 0.07728518 0.09997744 not norm

# test error
cd41.pred.rf <- predict(rffit,test.cd41)$prediction
mean.mspe.cd41 <- mean((test.cd41$y - cd41.pred.rf)^2)
sigma.mspe.cd41 <- 1/ntest * var((test.cd41$y - cd41.pred.rf)^2) 
c(mean.mspe.cd41 - 1.96 * sqrt(sigma.mspe.cd41), mean.mspe.cd41, mean.mspe.cd41 + 1.96 * sqrt(sigma.mspe.cd41))
#[1] 0.6516897 0.8301834 1.0086770
# 0.3332793 0.4620930 0.5909068 not norm

ggplot(data.frame(cbind(cd41.pred.rf,test.cd41$y)), aes(x = cd41.pred.rf, y = test.cd41$y))+
  geom_point() + ylim(-4,6) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'cd41 in test set vs predicted value - Random forest',x = 'Predicted value',y='True test value')
cor(cd41.pred.rf,test.cd41$y) #[1] 0.7452225 # 0.7454715 not norm
R2.rpart <- 1 - sum((test.cd41$y-cd41.pred.rf)^2)/sum((test.cd41$y-mean(test.cd41$y))^2)
R2.rpart #0.5191256 #[1] 0.5193228 not norm




















##########################pick 20 variables to train###############################

top14_importance.cd41 <- names(importance.rf)[1:13]
data.cd41_importance <- data.cd41[, colnames(data.cd41) %in% c('y',top14_importance.cd41)]
train.cd41_importance <- data.cd41_importance[strat_index,]
test.cd41_importance <- data.cd41_importance[-strat_index,]

rffit <-  ranger(formula = y ~ ., 
                 data = train.cd41_importance, 
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

cd41.pred.rf <- predict(rffit,test.cd41)$prediction

# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.cd41 <- mean((test.cd41$y - cd41.pred.rf)^2)
sigma.mspe.cd41 <- 1/ntest * var((test.cd41$y - cd41.pred.rf)^2) 
# 95% CI:
c(mean.mspe.cd41 - 1.96 * sqrt(sigma.mspe.cd41), mean.mspe.cd41, mean.mspe.cd41 + 1.96 * sqrt(sigma.mspe.cd41))
# [1] 0.3005706 0.4231725 0.5457743 top 30
# [1] 0.2659077 0.3794501 0.4929925 top 20
# [1] 0.2546193 0.3629273 0.4712352 top 15
# [1] 0.2462357 0.3609213 0.4756069 top 14

plot(cd41.pred.rf,test.cd41$y)
abline(a=0, b=1)
sum(cd41.pred.rf <= test.cd41$y) # 82
sum(cd41.pred.rf > test.cd41$y) # more overestimate 129
