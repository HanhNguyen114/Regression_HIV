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

#############
library(psych)
multi.hist(total_data[,2:7])
ggplot(total_data,aes(x = arm, fill = arm)) + geom_histogram(binwidth = 0.5, stat = "count") +
  scale_color_manual(values = c("red",'orange','yellow','green','dark blue','purple')) +
  labs(title = 'Distribution of patients number in each arm')
#############
library(naniar)
total_data.imputed <- total_data %>% bind_shadow(only_miss = T)
vis_miss(total_data.imputed[,(ncol(total_data)+1) : ncol(total_data.imputed)])
missing_var = c('pr6_NA','pr6',
                'pr7_NA','pr7',
                'pr8_NA','pr8',
                'pr9_NA','pr9',
                'pr99_NA','pr99',
                'rt38_NA','rt38',
                'rt39_NA','rt39',
                'rt40_NA','rt40',
                'rt238_NA','rt238',
                'rt239_NA','rt239',
                'rt240_NA','rt240')
total_data.imputed <- total_data.imputed[, colnames(total_data.imputed) %in% missing_var]
library(visdat)
vis_miss(total_data.imputed[,1:11])

sum(is.na(total_data))
sum(is.na(total_data.imputed[,1:11]))

total_data <- total_data %>% na.omit() #remove NA observations 879 - 85 = 24 obs with NA
#####################################################################################
# mean values of lrna1, ... among 6 arms
total_data.extract <- total_data[,2:8]
total_data.extract <- gather(total_data.extract, 'features', 'value', 1:6)

ggplot(total_data.extract[1:nrow(total_data.extract)/2,], aes(x = arm, y = value)) + 
  geom_boxplot() + facet_wrap(~ features) + labs(title = "Mean values of Lrna0, Lrna1 and Lrna2")
ggplot(total_data.extract[(nrow(total_data.extract)/2+1):nrow(total_data.extract),], aes(x = arm, y = value)) + 
  geom_boxplot() + facet_wrap(~ features) + labs(title = "Mean values of cd40, cd41 and cd42")
#####################################################################################
total_data <- total_data[, ! colnames(total_data) %in% c("patid")]
# 304 - 233 = 71 constant variables
total_data_normalize<- predict(preProcess(total_data[,1:3], method=c('YeoJohnson')), total_data)

library(psych)
total_data.processed <- predict(preProcess(total_data_normalize[,! colnames(total_data_normalize) %in% c('arm')], 
                                           method=c('zv','nzv',"center", "scale"))
                                ,total_data_normalize)
#total_data.processed <- predict(preProcess(total_data[,! colnames(total_data) %in% c('arm')], 
 #                                          method=c('zv','nzv',"center", "scale"))
  #                              ,total_data)
multi.hist(total_data.processed[,1:3])

comat <- cor(total_data.processed[, colnames(total_data.processed) != 'arm'])
hc = findCorrelation(comat, cutoff=0.9,exact = F) # put any value as a "cutoff" 
hc = sort(hc)
############################################################################################
# index for train and test split
# Stratify data
seed <- 42
set.seed(seed)
library(caret)
strat_index <- createDataPartition(y = total_data$arm,list = FALSE,p = 0.75)

###########################################################################################
# data for lrna1
data.lrna1 <- total_data.processed[, !colnames(total_data.processed) %in% c('lrna2','cd41','cd42')]
names(data.lrna1)[2] <- 'y'
train.lrna1 <- data.lrna1[strat_index,]
test.lrna1 <- data.lrna1[-strat_index,]

############################################################################################
#creat 5 folds for cv
num_cv_folds = 5
set.seed(seed)
cv_split_lrna1 <- vfold_cv(train.lrna1, v=num_cv_folds, strata = y)
cv_data_lrna1 <- cv_split_lrna1 %>% 
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
#RANDOM FORREST WITH RANGER PACKAGE WITH LRNA1
library(ranger)

# Prepare for tuning the cross validation folds BY mtry, nodesize and sampsize
mtry_min <- 1
mtry_max <- floor(sqrt((ncol(train.lrna1)-1)))
mtry <- seq(mtry_min,mtry_max,by =2)
nodesize <- seq(2,10,2)
numtree <- seq(300,800,100)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(mtry = mtry, nodesize = nodesize, numtree = numtree)

#CROSS THE ELEMENT OF HYPER GRID 
cv_tune_lrna1 <- cv_data_lrna1 %>%
  tidyr::crossing( mtry = hyper_grid$mtry,nodesize = hyper_grid$nodesize, numtree = hyper_grid$numtree) 


# Build RANDOM FOREST model for each fold & mtry/nodesize/sampsize combination 

cv_model_lrna1 <- cv_tune_lrna1 %>%
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
cv_model_lrna1 <- cv_model_lrna1 %>% 
  mutate(validate_predicted = map2(.x = model, .y = validate, ~ predict(.x, .y)$predictions),
         validate_actual = map(validate, ~ .x$y))  %>%
  # Calculate validate accuracy for each fold and mtry combination
  mutate(error = map2_dbl(.x = validate_actual, .y = validate_predicted, ~ mean((.x-.y)^2)))

cv_eval_lrna1 <- cv_model_lrna1 %>%
  group_by(numtree,mtry,nodesize) %>%
  summarise(mean_error = mean(error)) %>%
  ungroup(mtry,nodesize,numtree)


ggplot(data = cv_eval_lrna1, aes(x=nodesize,y = mean_error)) + geom_line() + 
  facet_wrap(~numtree,labeller = label_both)+
  labs(title ="Mean error by nodesize", x = "nodesize", y = "Error") +geom_smooth()


best_mtry = cv_eval_lrna1$mtry[cv_eval_lrna1$mean_error == min(cv_eval_lrna1$mean_error)]
best_nodesize = cv_eval_lrna1$nodesize[cv_eval_lrna1$mean_error == min(cv_eval_lrna1$mean_error)]
best_numtree = cv_eval_lrna1$numtree[cv_eval_lrna1$mean_error == min(cv_eval_lrna1$mean_error)]

rffit <-  ranger(formula = y ~ ., 
                 data = train.lrna1, 
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
ntest <- length(test.lrna1$y)

#train error
lrna1.pred.rf <- predict(rffit,train.lrna1)$prediction
plot(lrna1.pred.rf,train.lrna1$y)
lrna1.lm.train <- lm(train.lrna1$y~lrna1.pred.rf-1)
abline(a= 0, b=0)
abline(a= 0, b=lrna1.lm.train$coefficients[1])
ggplot(data.frame(cbind(lrna1.pred.rf,train.lrna1$y)), aes(x = lrna1.pred.rf, y = train.lrna1$y))+
  geom_point() + ylim(-3,3) + xlim(-1,1) +geom_abline(intercept = 0, slope = testlm$coefficients[1])+
  labs(title = 'lrna1 in train set vs predicted value - Random forest',x = 'Predicted value',y='True train value')

mean.mspe.lrna1 <- mean((train.lrna1$y - lrna1.pred.rf)^2)
sigma.mspe.lrna1 <- 1/ntest * var((train.lrna1$y - lrna1.pred.rf)^2) 
# 95% CI:
c(mean.mspe.lrna1 - 1.96 * sqrt(sigma.mspe.lrna1), mean.mspe.lrna1, mean.mspe.lrna1 + 1.96 * sqrt(sigma.mspe.lrna1))
#[1] 0.6865857 0.8166068 0.9466278 with norm
# 0.5734200 0.8013674 1.0293148 not norm 

# test error
lrna1.pred.rf <- predict(rffit,test.lrna1)$prediction

# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna1 <- mean((test.lrna1$y - lrna1.pred.rf)^2)
sigma.mspe.lrna1 <- 1/ntest * var((test.lrna1$y - lrna1.pred.rf)^2) 
# 95% CI:
c(mean.mspe.lrna1 - 1.96 * sqrt(sigma.mspe.lrna1), mean.mspe.lrna1, mean.mspe.lrna1 + 1.96 * sqrt(sigma.mspe.lrna1))
# [1] 0.8727424 1.0434586 1.2141747 with norm
# 0.832139 1.168299 1.504459 not norm
plot(lrna1.pred.rf,test.lrna1$y)
testlm <- lm(test.lrna1$y ~ lrna1.pred.rf -1)
abline(a= 0, b= testlm$coefficients[1])
sum(lrna1.pred.rf <= test.lrna1$y) # 62
sum(lrna1.pred.rf > test.lrna1$y) # more overestimate 

var(test.lrna1$y)
ggplot(data.frame(cbind(lrna1.pred.rf,test.lrna1$y)), aes(x = lrna1.pred.rf, y = test.lrna1$y))+
  geom_point() + xlim(-1,1) +geom_abline(intercept = 0, slope = testlm$coefficients[1])+
  labs(title = 'lrna1 in test set vs predicted value - Random forest',x = 'Predicted value',y='True test value')

cor(lrna1.pred.rf,test.lrna1$y) #[1] 0.1056126 # 0.0467153 not norm

R2.rpart <- 1 - sum((test.lrna1$y-lrna1.pred.rf)^2)/sum((test.lrna1$y-mean(test.lrna1$y))^2)
R2.rpart #-0.003752057 # -0.01025565 not norm



















##########################pick 20 variables to train###############################

top12_importance.lrna1 <- names(importance.rf)[1:12]
data.lrna1_importance <- data.lrna1[, colnames(data.lrna1) %in% c('y',top12_importance.lrna1)]
train.lrna1_importance <- data.lrna1_importance[strat_index,]
test.lrna1_importance <- data.lrna1_importance[-strat_index,]

rffit <-  ranger(formula = y ~ ., 
                 data = train.lrna1_importance, 
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

lrna1.pred.rf <- predict(rffit,test.lrna1)$prediction

# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.lrna1 <- mean((test.lrna1$y - lrna1.pred.rf)^2)
sigma.mspe.lrna1 <- 1/ntest * var((test.lrna1$y - lrna1.pred.rf)^2) 
# 95% CI:
c(mean.mspe.lrna1 - 1.96 * sqrt(sigma.mspe.lrna1), mean.mspe.lrna1, mean.mspe.lrna1 + 1.96 * sqrt(sigma.mspe.lrna1))


plot(lrna1.pred.rf,test.lrna1$y)
abline(a=0, b=1)
sum(lrna1.pred.rf <= test.lrna1$y) # 71
sum(lrna1.pred.rf > test.lrna1$y) # more overestimate 140
