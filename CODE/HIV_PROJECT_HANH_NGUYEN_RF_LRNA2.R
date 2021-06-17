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

#total_data$A <- ifelse(total_data$arm == "A",1,0)
#total_data$B <- ifelse(total_data$arm == "B",1,0)
#total_data$C <- ifelse(total_data$arm == "C",1,0)
#total_data$D <- ifelse(total_data$arm == "D",1,0)
#total_data$E <- ifelse(total_data$arm == "E",1,0)
#total_data$F <- ifelse(total_data$arm == "F",1,0)

total_data <- total_data[, ! colnames(total_data) %in% c("patid")]
# 304 - 233 = 71 constant variables
total_data_normalize<- predict(preProcess(total_data[,1:3], method=c('YeoJohnson')), total_data)

library(psych)
#total_data.processed <- predict(preProcess(total_data_normalize[,! colnames(total_data_normalize) %in% c('arm')], 
 #                                          method=c('zv','nzv',"center", "scale"))
  #                              ,total_data_normalize)
total_data.processed <- predict(preProcess(total_data[,! colnames(total_data_normalize) %in% c('arm')], 
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
# data for lrna2
data.lrna2 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna1','cd41','cd42')]
names(data.lrna2)[2] <- 'y'
train.lrna2 <- data.lrna2[strat_index,]
test.lrna2 <- data.lrna2[-strat_index,]
############################################################################################
#creat 5 folds for cv
num_cv_folds = 5
set.seed(seed)
cv_split_lrna2 <- vfold_cv(train.lrna2, v=num_cv_folds, strata = y)
cv_data_lrna2 <- cv_split_lrna2 %>% 
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
#RANDOM FORREST WITH RANGER PACKAGE WITH lrna2
library(ranger)

# Prepare for tuning the cross validation folds BY mtry, nodesize and sampsize
mtry_min <- 1
mtry_max <- floor(sqrt((ncol(train.lrna2)-1)))
mtry <- seq(mtry_min,mtry_max,by=2)
nodesize <- seq(2,10,2)
numtree <- seq(300,800,100)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, numtree = numtree)

#CROSS THE ELEMENT OF HYPER GRID 
cv_tune_lrna2 <- cv_data_lrna2 %>%
  tidyr::crossing( mtry = hyper_grid$mtry,nodesize = hyper_grid$nodesize, numtree = hyper_grid$numtree) 


# Build RANDOM FOREST model for each fold & mtry/nodesize/sampsize combination 

cv_model_lrna2 <- cv_tune_lrna2 %>%
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
cv_model_lrna2 <- cv_model_lrna2 %>% 
  mutate(validate_predicted = map2(.x = model, .y = validate, ~ predict(.x, .y)$predictions),
         validate_actual = map(validate, ~ .x$y))  %>%
  # Calculate validate accuracy for each fold and mtry combination
  mutate(error = map2_dbl(.x = validate_actual, .y = validate_predicted, ~ mean((.x-.y)^2)))

cv_eval_lrna2 <- cv_model_lrna2 %>%
  group_by(numtree,mtry,nodesize) %>%
  summarise(mean_error = mean(error)) %>%
  ungroup(mtry,nodesize,numtree)


ggplot(data = cv_eval_lrna2, aes(x=nodesize,y = mean_error)) + geom_line() + 
  facet_wrap(~numtree,labeller = label_both)+
  labs(title ="Mean error by nodesize", x = "nodesize", y = "Error") 


best_mtry = cv_eval_lrna2$mtry[cv_eval_lrna2$mean_error == min(cv_eval_lrna2$mean_error)]
best_nodesize = cv_eval_lrna2$nodesize[cv_eval_lrna2$mean_error == min(cv_eval_lrna2$mean_error)]
best_numtree = cv_eval_lrna2$numtree[cv_eval_lrna2$mean_error == min(cv_eval_lrna2$mean_error)]

rffit <-  ranger(formula = y ~ ., 
                 data = train.lrna2, 
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
ntest <- length(test.lrna2$y)
ntrain <- length(train.lrna2$y)
#train error
lrna2.pred.rf <- predict(rffit,train.lrna2)$prediction
plot(lrna2.pred.rf,train.lrna2$y)
lrna2.lm.train <- lm(train.lrna2$y~lrna2.pred.rf-1)
abline(a= lrna2.lm.train$coefficients[1], b=lrna2.lm.train$coefficients[2])
mean.mspe.lrna2 <- mean((train.lrna2$y - lrna2.pred.rf)^2)
sigma.mspe.lrna2 <- 1/ntest * var((train.lrna2$y - lrna2.pred.rf)^2) 
c(mean.mspe.lrna2 - 1.96 * sqrt(sigma.mspe.lrna2), mean.mspe.lrna2, mean.mspe.lrna2 + 1.96 * sqrt(sigma.mspe.lrna2))
ggplot(data.frame(cbind(lrna2.pred.rf,train.lrna2$y)), aes(x = lrna2.pred.rf, y = train.lrna2$y))+
  geom_point() + ylim(-3,3) + xlim(-1,1) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'lrna2 in train set vs predicted value - Random forest',x = 'Predicted value',y='True train value')
# NOT NORM 0.5848335 0.7847865 0.9847394 NOT NORM


# test error
lrna2.pred.rf <- predict(rffit,test.lrna2)$prediction
mean.mspe.lrna2 <- mean((test.lrna2$y - lrna2.pred.rf)^2)
sigma.mspe.lrna2 <- 1/ntest * var((test.lrna2$y - lrna2.pred.rf)^2) 
c(mean.mspe.lrna2 - 1.96 * sqrt(sigma.mspe.lrna2), mean.mspe.lrna2, mean.mspe.lrna2 + 1.96 * sqrt(sigma.mspe.lrna2))
# [1] 0.9067559 1.0685338 1.2303117
# NOT NORM 0.8152417 1.0600016 1.3047615

plot(lrna2.pred.rf,test.lrna2$y)
abline(a= 0, b= 1)
sum(lrna2.pred.rf <= test.lrna2$y) # 62
sum(lrna2.pred.rf > test.lrna2$y) # more overestimate 149
ggplot(data.frame(cbind(lrna2.pred.rf,test.lrna2$y)), aes(x = lrna2.pred.rf, y = test.lrna2$y))+
  geom_point() + ylim(-2,4) + xlim(-1,1) +geom_abline(intercept = 0, slope = 1)+
  labs(title = 'lrna2 in test set vs predicted value - Random forest',x = 'Predicted value',y='True test value')
cor(lrna2.pred.rf,test.lrna2$y) #[1] 0.06512775 # 0.06663725 NOT NORM

R2.rpart <- 1 - sum((test.lrna2$y-lrna2.pred.rf)^2)/sum((test.lrna2$y-mean(test.lrna2$y))^2)
R2.rpart #-0.02581952 # 0.06663725 NOT NORM













##########################pick 20 variables to train###############################

top20_importance.lrna2 <- names(importance.rf)[1:200]
data.lrna2_importance <- data.lrna2[, colnames(data.lrna2) %in% c('y',top20_importance.lrna2)]
train.lrna2_importance <- data.lrna2_importance[strat_index,]
test.lrna2_importance <- data.lrna2_importance[-strat_index,]

rffit <-  ranger(formula = y ~ ., 
                 data = train.lrna2_importance, 
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

lrna2.pred.rf <- predict(rffit,test.lrna2)$prediction

# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna2 <- mean((test.lrna2$y - lrna2.pred.rf)^2)
sigma.mspe.lrna2 <- 1/ntest * var((test.lrna2$y - lrna2.pred.rf)^2) 
# 95% CI:
c(mean.mspe.lrna2 - 1.96 * sqrt(sigma.mspe.lrna2), mean.mspe.lrna2, mean.mspe.lrna2 + 1.96 * sqrt(sigma.mspe.lrna2))
# [1] 0.8096555 1.0596049 1.3095544
plot(lrna2.pred.rf,test.lrna2$y)
abline(a=0, b=1)
sum(lrna2.pred.rf <= test.lrna2$y) # 68
sum(lrna2.pred.rf > test.lrna2$y) # more overestimate 143
