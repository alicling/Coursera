Prediction Assignment Writeup
================
Xiang Li
6/10/2017

### libray

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 3.3.3

    ## Loading required package: lattice

    ## Warning: package 'lattice' was built under R version 3.3.2

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.3.2

``` r
library(rpart.plot)
```

    ## Warning: package 'rpart.plot' was built under R version 3.3.3

    ## Loading required package: rpart

    ## Warning: package 'rpart' was built under R version 3.3.3

### read the data

#### read the data into R studio, create trainset and testset by dividing the pml-training data

``` r
train_all<-read.csv("./pml-training.csv")
train_all<-tbl_df(train_all)
inTrain  <- createDataPartition(train_all$classe, p=0.7, list=FALSE)
train <- train_all[inTrain, ]
test <- train_all[-inTrain, ]
```

### remove data with NAs and blanks

#### we remove the data that are not useful which includes the nas and blanks

``` r
train[train==""]<-NA
table(is.na(train))
```

    ## 
    ##   FALSE    TRUE 
    ##  851320 1346600

``` r
countna<-function(x){length(which(is.na(x)))}
nalist<-apply(train,2,countna)
nalist<-tbl_df(nalist)
nacol<-which(nalist>50)
train<-train[,-nacol]
test<-test[,-nacol]
table(is.na(train))
```

    ## 
    ##  FALSE 
    ## 824220

### remove unnessary data

#### the first seven columns are not that useful so we remove them

``` r
train<-train[,-c(1:7)]
test<-test[,-c(1:7)]
```

### Build up a model--use decision tree

#### we use rpart to build up a decision tree model

``` r
rpart <- rpart(classe ~., data = train, method="class")
rpart.plot(rpart)
```

    ## Warning: labs do not fit even at cex 0.15, there may be some overplotting

![](Coursera2_files/figure-markdown_github/unnamed-chunk-5-1.png) \#\#\# Confusion Matrix \#\#\#\# the decision tree model is not so accurate. the accuracy is 0.7361 and the kappa is 0.6646

``` r
predict1<- predict(rpart, newdata=test, type="class")
confusionMatrix(predict1,test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1520  178   33   47   16
    ##          B   47  657   51   67   82
    ##          C   38  116  805  130  118
    ##          D   21   74   68  627   51
    ##          E   48  114   69   93  815
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7517          
    ##                  95% CI : (0.7405, 0.7627)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6853          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9080   0.5768   0.7846   0.6504   0.7532
    ## Specificity            0.9349   0.9480   0.9173   0.9565   0.9325
    ## Pos Pred Value         0.8473   0.7268   0.6669   0.7455   0.7155
    ## Neg Pred Value         0.9624   0.9032   0.9528   0.9332   0.9437
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2583   0.1116   0.1368   0.1065   0.1385
    ## Detection Prevalence   0.3048   0.1536   0.2051   0.1429   0.1935
    ## Balanced Accuracy      0.9215   0.7624   0.8509   0.8035   0.8429

### build up a model--use random forest

#### then we try to build up a random forest model to get higher accuracy. and the accuracy is more than 98%, which is high enough for us to predict.

``` r
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
rf <- train(classe ~ ., data=train, method="rf",
                          trControl=controlRF)
```

    ## Loading required package: randomForest

    ## Warning: package 'randomForest' was built under R version 3.3.3

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

``` r
rf
```

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 9158, 9158, 9158 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9875519  0.9842509
    ##   27    0.9873335  0.9839731
    ##   52    0.9799811  0.9746720
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 2.

``` r
predict2 <- predict(rf, newdata=test)
confusionMatrix(predict2, test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    2    0    0    0
    ##          B    0 1137   11    0    0
    ##          C    0    0 1015   13    1
    ##          D    0    0    0  950    7
    ##          E    0    0    0    1 1074
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9941          
    ##                  95% CI : (0.9917, 0.9959)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9925          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9982   0.9893   0.9855   0.9926
    ## Specificity            0.9995   0.9977   0.9971   0.9986   0.9998
    ## Pos Pred Value         0.9988   0.9904   0.9864   0.9927   0.9991
    ## Neg Pred Value         1.0000   0.9996   0.9977   0.9972   0.9983
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1932   0.1725   0.1614   0.1825
    ## Detection Prevalence   0.2848   0.1951   0.1749   0.1626   0.1827
    ## Balanced Accuracy      0.9998   0.9980   0.9932   0.9920   0.9962

### Predict

#### since the random forest model is more accurate, so we use this model to predict

``` r
newtest<-read.csv("./pml-testing.csv")
predict(rf, newdata=newtest)
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
