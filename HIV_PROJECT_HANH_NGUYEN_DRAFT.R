source('HIVDATA.R')

library(dplyr)
library(ggplot2)
library(forecast)
library(caret) #train test split
library(rsample) #vfold
library(purrr) #map
library(e1071) # SVM
library(svrpath) #lasso L1 penalty 


# Explore the data
dat384.summary <- dat384 %>% group_by(arm) %>% summarize(meanL0=mean(lrna0),
                                                         meanL1=mean(lrna1),
                                                         meanL2=mean(lrna2),
                                                         meanC0=mean(cd40),
                                                         meanC1=mean(cd41),
                                                         meanC2=mean(cd42))
#print(dat384.summary)


#ggplot(data = dat384,aes(x = arm)) + geom_bar()
#ggplot(data = dat384.summary,aes(x = arm, y= meanL0)) + geom_col()
#ggplot(data = dat384.summary,aes(x = arm, y= meanL1)) + geom_col()
#ggplot(data = dat384.summary,aes(x = arm, y= meanL2)) + geom_col()
#ggplot(data = dat384.summary,aes(x = arm, y= meanC0)) + geom_col()
#ggplot(data = dat384.summary,aes(x = arm, y= meanC1)) + geom_col()
#ggplot(data = dat384.summary,aes(x = arm, y= meanC2)) + geom_col()

#ggplot(data=dat384,aes(x=lrna0))+geom_histogram(stat="bin",binwidth=0.3)+facet_wrap(~arm)
#ggplot(data=dat384,aes(x=log(lrna1)))+geom_histogram(stat="bin",binwidth=0.3)+facet_wrap(~arm)
#ggplot(data=dat384,aes(x=lrna2))+geom_histogram(stat="bin",binwidth=0.3)+facet_wrap(~arm)
#ggplot(data=dat384,aes(x=cd40))+geom_histogram(stat="bin",binwidth=80)+facet_wrap(~arm)
#ggplot(data=dat384,aes(x=cd41))+geom_histogram(stat="bin",binwidth=80)+facet_wrap(~arm)
#ggplot(data=dat384,aes(x=cd42))+geom_histogram(stat="bin",binwidth=80)+facet_wrap(~arm)



###########################################################################################
# Creat total dataset for features
basedata <- merge(prmut0,rtmut0,by = "patid")

###########################################################################################
# Total data
data<- data.frame(cbind(dat384,basedata))
data <- data %>% na.omit()
sum(is.na(data))

# normalize data using boxcox

#trans.lrna1 <- boxCoxVariable(data$lrna1)


data$lrna0 <- forecast::BoxCox( data$lrna0, BoxCox.lambda(data$lrna0))

data$lrna1 <- forecast::BoxCox( data$lrna1, BoxCox.lambda(data$lrna1))

data$lrna2 <- forecast::BoxCox( data$lrna2, BoxCox.lambda(data$lrna2))

data$cd40 <- forecast::BoxCox( data$cd40, BoxCox.lambda(data$cd40))

data$cd41 <- forecast::BoxCox( data$cd41, BoxCox.lambda(data$cd41))

data$cd42 <- forecast::BoxCox( data$cd42, BoxCox.lambda(data$cd42))

#remove zero-variance variables
zeroVar <- function(data, useNA = 'ifany') {
  out <- apply(data, 2, function(x) {length(table(x, useNA = useNA))})
  which(out==1) }
data <- data[,-zeroVar(data)]

# Center and scale data
data <- data.frame(cbind('patid'=data$patid,cbind('arm'=data$arm,scale(data[, !names(data) %in% c('patid','arm')],center=T,scale=T))))

##############################################################################
##############################################################################
# DATA FOR LRNA1
# Remove the useless "id" column
unused_var <- c('patid','patid.1','lrna1','lrna2','cd41','cd42')
data_temp <- data[, !names(data) %in% unused_var]
dat_lrna1 <- cbind('y' = data$lrna1,data_temp)
summary(dat_lrna1$y)
summary(dat384$lrna1)

# Stratify data
seed <- 42
set.seed(seed)
library(caret)
strat_index <- createDataPartition(y = data$arm,list = FALSE,p = 0.75)

#Split train and test
train.lrna1 <- dat_lrna1[strat_index,]
test.lrna1 <- dat_lrna1[-strat_index,]

ggplot(data = train.lrna1,aes(x = arm)) + geom_bar()
ggplot(data = test.lrna1,aes(x = arm)) + geom_bar()

##################################################################################
##################################################################################
#creat 5 folds for cv
num_cv_folds = 5
set.seed(seed)
cv_split_lrna1 <- vfold_cv(train.lrna1, v=num_cv_folds,strata = y)
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
#RANDOM FORREST WITH RANGER PACKAGE
library(ranger)
# Prepare for tuning the cross validation folds BY mtry, nodesize and sampsize
#mtry_min <- 1
#mtry_max <- floor(sqrt((ncol(train.lrna1)-1)))
mtry <- 1
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
  mutate(validate_predicted = map2(.x = model, .y = validate, ~predict(.x, .y)$predictions),
         validate_actual = map(validate, ~ .x$y))  %>%
  # Calculate validate accuracy for each fold and mtry combination
  mutate(error = map2_dbl(.x = validate_actual, .y = validate_predicted, ~sqrt(sum((.x-.y)^2))))

cv_eval_lrna1 <- cv_model_lrna1 %>%
  group_by(numtree,mtry,nodesize) %>%
  summarise(mean_error = mean(error)) %>%
  ungroup(mtry,nodesize,numtree)


ggplot(data = cv_eval_lrna1, aes(x=nodesize,y = mean_error)) + geom_line() + 
  facet_wrap(~numtree,labeller = label_both)+
  labs(title ="Mean error by nodesize", x = "nodesize", y = "Error") +geom_smooth()

best_mtry = cv_model_lrna1$mtry[cv_model_lrna1$error == min(cv_model_lrna1$error)]
best_nodesize = cv_model_lrna1$nodesize[cv_model_lrna1$error == min(cv_model_lrna1$error)]
best_numtree = cv_model_lrna1$numtree[cv_model_lrna1$error == min(cv_model_lrna1$error)]

rffit <-  ranger(formula = y ~ ., 
                 data = train.lrna1, 
                 mtry = best_mtry, 
                 num.trees = best_numtree, 
                 min.node.size = best_nodesize,
                 seed = seed, 
                 oob.error = TRUE,
                 importance = 'impurity',
                 classification = FALSE,
                 replace = TRUE,
                 verbose = TRUE,
                 respect.unordered.factors = FALSE,
                 regularization.factor = 1)
rffit$prediction.error
rffit$r.squared
importance.rf <- sort(rffit$variable.importance,decreasing = T)
ntest <- length(test.lrna1$y)
predict.result <- data.frame('y' = test.lrna1$y)
predict.result$predicted_y.rf <- predict(rffit, test.lrna1)$predictions

plot(predict.result$predicted_y.rf,test.lrna1$y)
abline(a=0, b=1)
error.rf <- sqrt(sum((predict.result$y - predict.result$predicted_y.rf)^2)/length(test.lrna1$y))
#before respect and regu 0.955516323169075
######################################################################################
######################################################################################
# CREATE SVM REGRESSION


para.obj <- tune(svm, y ~.,data = train.lrna1,
                kernel = "polynomial",
                ranges=list (gamma = c(2,1,0.1,0.01,0.005),
                             degree = c(2,3,4,5,6),
                             cost = c(0.1,0.5)),
                tunecontrol = tune.control(sampling = "cross", cross = 5,
                                           best.model = T, performances =T))
para.obj$best.parameters

rmlist <- c('pr55','rt73','rt91','rt125','rt136','rt140','rt154','rt155','rt156','rt216','rt230')
svmfit <- svm(y ~ ., data = train.lrna1[, !colnames(train.lrna1) %in% rmlist], 
              kernel = 'polynomial',
              cost = 0.1, 
              gamma = 0.005,
              degree = 2)
predict.svm <- predict(svmfit,test.lrna1[,-1])
#predict.result$predicted_y.svm <- predict(svmfit,test.lrna1[,-1])
#0.965033384489204 poly, cost 0.1 gamma 0.01 degree 6

error.svm <- sqrt(sum((predict.result$y - predict.result$predicted_y.svm)^2)/length(test.lrna1$y))

plot(predict.result$predicted_y.svm,test.lrna1$y)

abline(a=0, b=1)

library(svrpath)
svr.regu <- svrpath(x = as.matrix(train.lrna1[,-1]), y = train.lrna1$y,
                    param.kernel = 2, ridge = 0.05, 
                    eps = 1e-08, 
                    lambda.min = 1e-08)
predict(svr.regu, as.matrix(test.lrna1[,-1]),lambda = NULL, criterion = "gacv")
######################################################################################
######################################################################################
# CREATE LINEAR REGRESSION
# ordinary

lm1 <- lm(y ~ .,data= train.lrna1)
predict.result$predicted_y.lm <- predict(lm1, test.lrna1[,-1])

plot(predict.result$predicted_y.lm,test.lrna1$y)
abline(a = 0, b= 1)
bfit<-step(lm1,direction="backward",trace=T,k=log(dim(test.lrna1)[1]))

importance.rf[1:5]
#lrna0      cd40       arm     rt200      pr14 
#0.5072584 0.4188081 0.3177797 0.2705009 0.1868881

#Step:  AIC=12.72
#y ~ lrna0 + pr98 + rt200 + rt234 + rt235

#Df Sum of Sq    RSS    AIC
#<none>               624.90 12.721
#- rt235  1    5.3189 630.22 12.828
#- pr98   1    5.9536 630.85 13.476
#- rt200  1    6.4724 631.37 14.005
#- rt234  1    6.5644 631.46 14.099
#- lrna0  1   12.3987 637.30 20.022

lm2 <- lm(y ~ lrna0 + pr98 + rt200 + rt234 + rt235,data= train.lrna1)
predict.result$predicted_y.lm2 <- predict(lm2, test.lrna1[,-1])
plot(test.lrna1$predicted_y.lm2,test.lrna1$y)
abline(a = 0, b= 1)


# COMPARE WITH NON CENTER NO SCALING NO TRANSFORMATION
# Total data
data2<- data.frame(cbind(dat384,basedata))
data2 <- data2 %>% na.omit()
sum(is.na(data2))

data2 <- data2[,-zeroVar(data2)]

# DATA FOR LRNA1
# Remove the useless "id" column
unused_var <- c('patid','patid.1','lrna1','lrna2','cd41','cd42')
data_temp <- data2[, !names(data) %in% unused_var]
dat_lrna1.2 <- cbind('y' = data2$lrna1,data_temp)

#Split train and test
train.lrna1.2 <- dat_lrna1.2[strat_index,]
test.lrna1.2 <- dat_lrna1.2[-strat_index,]

lm3 <- lm(y ~ .,data= train.lrna1.2)
predict.result$predicted_y.lm3 <- predict(lm3, test.lrna1.2[,-1])

bfit<-step(lm3,direction="backward",trace=T,k=log(dim(test.lrna1.2)[1]))
######################################################################################
######################################################################################
error.lm <- sqrt(sum((predict.result$y - predict.result$predicted_y.lm)^2)/length(test.lrna1$y))
error.lm2 <- sqrt(sum((predict.result$y - predict.result$predicted_y.lm2)^2)/length(test.lrna1$y))
error.rf <- sqrt(sum((predict.result$y - predict.result$predicted_y.rf)^2)/length(test.lrna1$y))
error.svm <- sqrt(sum((predict.result$y - predict.result$predicted_y.svm)^2)/length(test.lrna1$y))
error.lm3 <- sqrt(sum((test.lrna1.2$y - predict.result$predicted_y.lm3)^2)/length(test.lrna1.2$y))
