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
# Exploring data

sum(is.na(total_data))
glimpse(total_data)

#remove zv variables. 305 - 234 = 71 zv var
total_data.processed <- predict(preProcess(total_data,method = c('zv')),total_data)

#run a correlation and drop the insignificant ones
comat <- cor(total_data.processed[, !colnames(total_data.processed) %in% c('arm','patid')])
corr <- cor(comat)
#prepare to drop duplicates and correlations of 1    
corr[lower.tri(corr,diag = TRUE)] <- NA
# drop perfect correlation
corr[corr == 1] <- NA
#turn into a 3-column table
corr <- as.data.frame(as.table(corr))
#remove the NA values from above 
corr <- na.omit(corr) 

#select significant values  
corr <- subset(corr, abs(Freq) > 0.6) 
#sort by highest correlation
corr <- corr[order(-abs(corr$Freq)),] 
#turn corr back into matrix in order to plot with corrplot
mtx_corr <- reshape2::acast(corr, Var1~Var2, value.var="Freq")
#plot correlations visually

library(corrplot)
corrplot(mtx_corr, is.corr=FALSE, tl.col="black", na.label=" ")

#####################################################################################
#####################################################################################
#####################################################################################

total_data.processed2 <- predict(preProcess(total_data[,! colnames(total_data) %in% c('arm','patid')],
                                            method = c('zv','center','scale')),
                                 total_data)
total_data.processed2 <- total_data.processed2[, ! colnames(total_data.processed2) %in% c("patid")]


############################################################################################
# index for train and test split
# Stratify data
seed <- 42
set.seed(seed)
library(caret)
strat_index <- createDataPartition(y = total_data$arm,list = FALSE,p = 0.75)

###########################################################################################
# data for cd41
data.cd41 <- total_data.processed2[, !colnames(total_data.processed2) %in% c('lrna2','lrna1','cd42')]
names(data.cd41)[3] <- 'y'
train.cd41 <- data.cd41[strat_index,]
test.cd41 <- data.cd41[-strat_index,]
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# glmnet with elastic net

library(glmnet)
nvar <- ncol(train.cd41)-1

# maintain the same folds across all models
fold_id <- sample(1:10, size = length(train.cd41$y), replace=TRUE)

# search across a range of alphas
tuning_grid <- tibble::tibble(
  alpha      = seq(0, 1, by = .1),
  mse_min    = NA,
  mse_1se    = NA,
  lambda_min = NA,
  lambda_1se = NA
)
xmat.train.cd41 <- model.matrix(~ -1 + ., train.cd41[,colnames(train.cd41) != 'y'])
xmat.test.cd41 <- model.matrix(~ -1 + ., test.cd41[,colnames(test.cd41) != 'y'])
for(i in seq_along(tuning_grid$alpha)) {
  
  # fit CV model for each alpha value
  fit <- cv.glmnet(xmat.train.cd41, train.cd41$y, 
                   alpha = tuning_grid$alpha[i], foldid = fold_id)
  
  # extract MSE and lambda values
  tuning_grid$mse_min[i]    <- fit$cvm[fit$lambda == fit$lambda.min]
  tuning_grid$mse_1se[i]    <- fit$cvm[fit$lambda == fit$lambda.1se]
  tuning_grid$lambda_min[i] <- fit$lambda.min
  tuning_grid$lambda_1se[i] <- fit$lambda.1se
}

tuning_grid %>%
  mutate(se = mse_1se - mse_min) %>%
  ggplot(aes(alpha, mse_min)) +
  geom_line(size = 2) +
  geom_ribbon(aes(ymax = mse_min + se, ymin = mse_min - se), alpha = .25) +
  ggtitle("MSE ± one standard error")

best_alpha = tuning_grid$alpha[which.min(tuning_grid$mse_min)]


# fit cv model to find lambda
cv_net <- cv.glmnet(xmat.train.cd41, 
                    train.cd41$y, alpha = best_alpha)
plot(cv_net$lambda,cv_net$cvm)

#glmnet.mod <- glmnet(as.matrix(trainset[,-ncol(trainset)]), 
#                    trainset$class, alpha = best_alpha, family = 'binomial')

cd41.pred.glmnet <- predict(cv_net, s = cv_net$lambda.min, type="response",xmat.test.cd41)

ntest <- length(test.cd41$y)
# psi is the mean squared prediction error (MSPE) estimate
# sigma2 is the estimate of the variance of the MSPE
mean.mspe.cd41 <- mean((test.cd41$y - cd41.pred.glmnet)^2)
sigma.mspe.cd41 <- 1/ntest * var((test.cd41$y - cd41.pred.glmnet)^2) 
# 95% CI:
c(mean.mspe.cd41 - 1.96 * sqrt(sigma.mspe.cd41), mean.mspe.cd41, mean.mspe.cd41 + 1.96 * sqrt(sigma.mspe.cd41))
# [1] 0.2625355 0.3887042 0.5148728
plot(cd41.pred.glmnet,test.cd41$y)
abline(a= 0, b= 1)

