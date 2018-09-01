#### Loading required libraries
library(dplyr)
library(cluster)
library(stringr)
library(factoextra)
library(caret)
library(rpart)
library(rpart.plot)
library(DMwR)
library(ROSE)
library(ebmc)
library(rusboost)
library(gridExtra)
library(e1071)
library(nnet)
library(corrplot)

# Import CFPB Consumer dataset 
data <- read.csv("ConsumerComplaintsAlltab.csv", sep = "\t")
str(data)

#remove complaint text, Tags, consumer_consent, submitted_via.
data_nt <- data[-c(6,11,12,13)]
str(data_nt)

#Demographics dataset
demo_tx <- read.csv("WebScraping/ZipTX/zip_tx_v1.csv")
demo_tx_na <- na.omit(demo_tx)
names(demo_tx_na)[1] <- "ZIP.code"
summary(demo_tx_na)


# Clustering on demographic dataset
demo_tx_na_rn <- demo_tx_na[,-1]
demo_tx_scale <- demo_tx_na[,-1]
demo_tx_scale <- scale(demo_tx_scale)
rownames(demo_tx_scale) <- demo_tx_na[,1]
demo_tx_eucl <- dist(demo_tx_scale, method = "euclidean")

#Scree plot
fviz_nbclust(demo_tx_scale, kmeans, method = "wss") + geom_vline(xintercept = 3, linetype = 2)

#K means Clustering
set.seed(123)
km.res <- kmeans(demo_tx_scale, 3, nstart = 50)
demo_tx_clust <- cbind(demo_tx_na_rn, cluster=km.res$cluster)

#Visualizing different cluster sizes
km.res.2 <- kmeans(demo_tx_scale, 2, nstart = 50)
km.res.3 <- kmeans(demo_tx_scale, 3, nstart = 50)
km.res.4 <- kmeans(demo_tx_scale, 4, nstart = 50)
km.res.5 <- kmeans(demo_tx_scale, 5, nstart = 50)
p1 <- fviz_cluster(km.res.2, data = demo_tx_scale, geom = "point")
p2 <- fviz_cluster(km.res.3, data = demo_tx_scale, geom = "point")
p3 <- fviz_cluster(km.res.4, data = demo_tx_scale, geom = "point")
p4 <- fviz_cluster(km.res.5, data = demo_tx_scale, geom = "point")
grid.arrange(p1, p2, p3, p4, nrow = 2)

#Cluster statistics
aggregate(demo_tx_na_rn, by = list(cluster=km.res$cluster), mean)
demo_tx_clust <- cbind(demo_tx_na_rn, cluster=km.res$cluster)

#cluster size
km.res$size

#cluster means
km.res$centers

#visualization
fviz_cluster(km.res, data = demo_tx_scale, geom = "point")

#Merge with original TEXAS dataset
data_zip_merge_tx <- data_nt %>% filter(data_nt$State=="TX")
demo_tx_zipclust_prep <- cbind(ZIP.code=demo_tx_na$ZIP.code, demo_tx_na_rn, cluster=km.res$cluster)
merged_clust_tx <- merge(data_zip_merge_tx, demo_tx_zipclust_prep, by="ZIP.code", all.x = T)
merged_clust_tx_na <- merged_clust_tx %>% filter(!is.na(cluster))
str(merged_clust_tx_na)

# Data preparation for predicting company response

#remove date, public response, state, date sent, consumer disputation, complaint id
tx_clust <- merged_clust_tx_na[-c(7, 9, 10, 12, 13, 31)]
tx_clust[,"Month"] <- as.factor(format(as.Date(tx_clust$Date.received, format="%m/%d/%Y"),"%m"))
#tx_clust[,"Year"] <- as.factor(format(as.Date(tx_clust$Date.received, format="%m/%d/%Y"),"%Y"))

#remove data received
tx_clust <-tx_clust[-2]
str(tx_clust)

#replace sub issue
for(i in 2:5){
  tx_clust[,i] <- as.character(tx_clust[,i])
}
x1 <- tx_clust$Sub.product==""
x2 <- tx_clust$Sub.issue==""

tx_clust$Sub.product[x1] <- tx_clust$Product[x1]
tx_clust$Sub.issue[x2] <- tx_clust$Issue[x2]

for(i in 2:5){
  tx_clust[,i] <- as.factor(tx_clust[,i])
}

#combine outcome categories
tx_clust$Company.response.to.consumer <-as.character(tx_clust$Company.response.to.consumer)

x1 <- tx_clust$Company.response.to.consumer == "Closed"
tx_clust$Company.response.to.consumer[x1] = "Consumer_fault"

x1 <- tx_clust$Company.response.to.consumer=="Closed with explanation"
tx_clust$Company.response.to.consumer[x1] = "Consumer_fault"

x1 <- tx_clust$Company.response.to.consumer=="Closed without relief"
tx_clust$Company.response.to.consumer[x1] = "Consumer_fault"

x1 <- tx_clust$Company.response.to.consumer == "Closed with monetary relief"
tx_clust$Company.response.to.consumer[x1] = "Company_fault"

x1 <- tx_clust$Company.response.to.consumer=="Closed with non-monetary relief"
tx_clust$Company.response.to.consumer[x1] = "Company_fault"

x1 <- tx_clust$Company.response.to.consumer=="Closed with relief"
tx_clust$Company.response.to.consumer[x1] = "Company_fault"

x1 <- tx_clust$Company.response.to.consumer == "Company_fault" | tx_clust$Company.response.to.consumer == "Consumer_fault"
tx_clust$Company.response.to.consumer[!x1] = "Late_Response"

final_tx_resp_na <- tx_clust %>% filter(tx_clust$Company.response.to.consumer!="Late_Response")

final_tx_resp_na$Company.response.to.consumer <-as.factor(final_tx_resp_na$Company.response.to.consumer)

#-------------------------------------------------------------------------------------Mortgage data
tx_clust_rep_mortgage <- final_tx_resp_na %>% filter(Product=="Mortgage")
tx_clust_rep_mortgage_issues <- tx_clust_rep_mortgage %>% filter(Issue=="Loan modification,collection,foreclosure" | Issue=="Loan servicing, payments, escrow account")

final_tx_resp_na_mortgage <- tx_clust_rep_mortgage_issues[-c(2)]

for(i in c(1,2,3,4,5,6,24)){
  final_tx_resp_na_mortgage[,i]<-as.character(final_tx_resp_na_mortgage[,i])
  final_tx_resp_na_mortgage[,i]<-as.factor(final_tx_resp_na_mortgage[,i])
}


## Modeling

#Remove complaint ID
final_tx <- final_tx_resp_na_mortgage[,-7] #final_tx_prep[-c(1,2, 7, 10, 14, 15)]
str(final_tx)
for(i in c(1,2,3,4,5,6,23)){
  final_tx[,i]<-as.character(final_tx[,i])
  final_tx[,i]<-as.factor(final_tx[,i])
}

for(i in 7:22){
  final_tx[,i]<-as.numeric(final_tx[,i])
}
final_tx[,7:22] <- scale(final_tx[,7:22])

#-----------------------------------------------------------------------------------------data exports
texas_mortgage <- final_tx_resp_na_mortgage[,-7] 
#write.csv(texas_mortgage, "texas_mortgage.csv")

ref_final_tx <- final_tx

#correlation
x <- final_tx
for(i in 1:ncol(x)){
  x[,i] <- as.numeric(x[,i])
}
x <- scale(x)
#write.csv(cor(x), "Correlation.csv")

#Excluding multi collinear variables
final_tx <- final_tx[,-c(1, 4, 7, 11, 12, 14, 15, 19)]

#----------------------------------------------------------------------------------------Train and test
set.seed(123)
partition<-createDataPartition(final_tx$Company.response.to.consumer,p=0.80,list = FALSE)
train_tx<-final_tx[partition,]
test_tx<-final_tx[-partition,]


#-------------------------------------------------------------------decision tree model

tx_model_resp <- rpart(Company.response.to.consumer ~ ., data=train_tx, method = "class")

#rpart.plot(tx_model_resp,digits = 2, split.fun=split.fun, faclen = 3)
#rpart.plot(tx_model_resp,digits = 2,fallen.leaves = TRUE,type = 3,extra = 1)

varImp(tx_model_resp)

wp1 <- predict(tx_model_resp,test_tx, type = "class")

confusionMatrix(wp1, test_tx$Company.response.to.consumer, positive = "Company_fault", mode = "everything")

roc.curve(test_tx$Company.response.to.consumer, wp1, plotit = F)

#----------------------------------------------------------------------OVer sampling

train_over <- ovun.sample(Company.response.to.consumer ~ ., data=train_tx, method = "over", N=9000, seed=123)$data

table(train_over$Company.response.to.consumer)

tx_model_resp <- rpart(Company.response.to.consumer ~ ., data=train_over, method = "class")

#rpart.plot(tx_model_resp,digits = 2, split.fun=split.fun, faclen = 3)

varImp(tx_model_resp)

wp1 <- predict(tx_model_resp,test_tx, type = "class")

confusionMatrix(wp1, test_tx$Company.response.to.consumer, positive = "Company_fault", mode = "everything")

roc.curve(test_tx$Company.response.to.consumer, wp1, plotit = F)

#accuracy.meas(test_tx$Company.response.to.consumer, wp1[,2])

#roc.curve(test_tx$Company.response.to.consumer, wp1, plotit = F)


#----------------------------------------------------------------------Undersampling 


train_under <- ovun.sample(Company.response.to.consumer ~ ., data=train_tx, method = "under", N=972, seed=123)$data

table(train_under$Company.response.to.consumer)

tx_model_resp <- rpart(Company.response.to.consumer ~ ., data=train_under, method = "class")

#rpart.plot(tx_model_resp,digits = 2, split.fun=split.fun, faclen = 3)

varImp(tx_model_resp)

wp1 <- predict(tx_model_resp,test_tx, type = "class")

confusionMatrix(wp1, test_tx$Company.response.to.consumer, positive = "Company_fault")

roc.curve(test_tx$Company.response.to.consumer, wp1, plotit = F)

#accuracy.meas(test_tx$Company.response.to.consumer, wp1[,2])

#roc.curve(test_tx$Company.response.to.consumer, wp1[,2], plotit = F)

#----------------------------------------------------------------------Both 

train_both <- ovun.sample(Company.response.to.consumer ~ ., data=train_tx, method = "both", p=0.5, seed = 123)$data
table(train_both$Company.response.to.consumer)

tx_model_resp <- rpart(Company.response.to.consumer ~ ., data=train_both, method = "class")

#rpart.plot(tx_model_resp,digits = 2, faclen = 3)

varImp(tx_model_resp)

wp1 <- predict(tx_model_resp,test_tx, type = "class")

confusionMatrix(wp1, test_tx$Company.response.to.consumer, positive = "Company_fault", mode="everything")

roc.curve(test_tx$Company.response.to.consumer, wp1, plotit = F)

#accuracy.meas(test_tx$Company.response.to.consumer, wp1[,2])

#roc.curve(test_tx$Company.response.to.consumer, wp1[,2], plotit = F)


#----------------------------------------------------------------------ROSE 
train_rose <- ROSE(Company.response.to.consumer ~ ., data=train_tx, seed = 123)$data

table(train_rose$Company.response.to.consumer)

tx_model_resp <- rpart(Company.response.to.consumer ~ ., data=train_rose, method = "class")

#rpart.plot(tx_model_resp,digits = 2, split.fun=split.fun, faclen = 3)

varImp(tx_model_resp)

wp1 <- predict(tx_model_resp,test_tx, type = "class")

confusionMatrix(wp1, test_tx$Company.response.to.consumer, positive = "Company_fault")

roc.curve(test_tx$Company.response.to.consumer, wp1, plotit = F)

#accuracy.meas(test_tx$Company.response.to.consumer, wp1[,2])

#roc.curve(test_tx$Company.response.to.consumer, wp1[,2], plotit = F)


#------------------------------------------------------------------------------DMwR SMOTE

set.seed(123)
train_SMOTE <- SMOTE(Company.response.to.consumer ~ ., data = train_tx, perc.over = 100, perc.under = 200)
table(train_SMOTE$Company.response.to.consumer)

tx_model_resp <- rpart(Company.response.to.consumer ~ ., data=train_SMOTE, method = "class")

#rpart.plot(tx_model_resp,digits = 2, split.fun=split.fun, faclen = 3)

varImp(tx_model_resp)

wp1 <- predict(tx_model_resp,test_tx, type = "class")

confusionMatrix(wp1, test_tx$Company.response.to.consumer, positive = "Company_fault")

roc.curve(test_tx$Company.response.to.consumer, wp1, plotit = F)


#accuracy.meas(test_tx$Company.response.to.consumer, wp1[,2])

#roc.curve(test_tx$Company.response.to.consumer, wp1[,2], plotit = F)


# SVM
tx_model_resp <- svm(Company.response.to.consumer ~ ., data=train_tx)

#varImp(tx_model_resp)

wp1 <- predict(tx_model_resp,test_tx, type = "class")

confusionMatrix(wp1, test_tx$Company.response.to.consumer, positive = "Company_fault")

roc.curve(test_tx$Company.response.to.consumer, wp1, plotit = F)

#nnet
num_tx_mortgage <- final_tx
num_tx_mortgage_y <- num_tx_mortgage[4]
num_tx_mortgage_x <- num_tx_mortgage[,-4]

for(i in (1:ncol(num_tx_mortgage_x))){
  num_tx_mortgage_x[,i] <- as.numeric(num_tx_mortgage_x[,i])
}

num_tx_mortgage_x <- scale(num_tx_mortgage_x)
num_tx <- cbind(num_tx_mortgage_x, num_tx_mortgage_y)

set.seed(123)
partition<-createDataPartition(num_tx$Company.response.to.consumer,p=0.70,list = FALSE)
train_tx<-num_tx[partition,]
test_tx<-num_tx[-partition,]
train_SMOTE <- SMOTE(Company.response.to.consumer ~ ., data = train_tx, perc.over = 100, perc.under = 200)
table(train_SMOTE$Company.response.to.consumer)
train_over <- ovun.sample(Company.response.to.consumer ~ ., data=train_tx, method = "over", N=9000, seed=123)$data
table(train_over$Company.response.to.consumer)

tx_model_resp <- nnet(Company.response.to.consumer ~ ., data=train_SMOTE, size = 10)

#varImp(tx_model_resp)

wp1 <- predict(tx_model_resp,test_tx, type = "class")

confusionMatrix(as.factor(wp1), test_tx$Company.response.to.consumer, positive = "Company_fault")

roc.curve(test_tx$Company.response.to.consumer, as.factor(wp1), plotit = F)

tx_model_resp <- rpart(Company.response.to.consumer ~ ., data=train_over, method = "class")

#rpart.plot(tx_model_resp,digits = 2, split.fun=split.fun, faclen = 3)

varImp(tx_model_resp)

wp1 <- predict(tx_model_resp,test_tx, type = "class")

confusionMatrix(wp1, test_tx$Company.response.to.consumer, positive = "Company_fault")

roc.curve(test_tx$Company.response.to.consumer, wp1, plotit = F)
