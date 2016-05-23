
# read data
digits = read.csv("/Users/PullingCarrot/Desktop/201509-12/STA141statisticalComputing/homework/hw3/digitsTrain.csv")
dim(digits)
head(digits)
summary(digits)

# randomilize the whole data
set.seed(0112)
randIndex = sample(1:nrow(digits), nrow(digits))
digits = digits[randIndex,]


# 1. euclidean distance matrix
# exclude the first label column when computing distance
eucD = dist(digits[,-1], method = "euclidean", diag = TRUE, upper = TRUE, p = 2)
eucD = as.matrix(eucD)
dim(eucD)
eucD[1:5, 1:5]

# 2. manhattan distance matrix
manD = dist(digits[,-1], method = "manhattan", diag = TRUE, upper = TRUE)
manD = as.matrix(manD)
dim(manD)
manD[1:5,1:5]

# 3. euclidean distance matrix after standardization for each pixel
# find out the columns with all zeros and exclude them when standardizing
temp = unlist(lapply(digits[,-1], function(column) round(max(column),5) > 0))
notAllZeroPixel = digits[,-1][,temp]
dim(notAllZeroPixel)
# standardize each of 784 pixels
stand = as.data.frame(lapply(notAllZeroPixel, function(x) (x-mean(x))/sd(x)))
dim(stand)
eucStandD = dist(stand, method = "euclidean", diag = TRUE, upper = TRUE)
eucStandD = as.matrix(eucStandD)
dim(eucStandD)
eucStandD[1:5, 1:5]

# 4. manhattan distance matrix after standardization for each pixel
manStandD = dist(stand, method = "manhattan", diag = TRUE, upper = TRUE)
manStandD = as.matrix(manStandD)
dim(manStandD)
manStandD[1:5, 1:5]

# 5. euclidean distance matrix after scaling by maximum value
scale = as.data.frame(lapply(notAllZeroPixel, function(x) x/max(x)))
eucScaleD = dist(scale, method = "euclidean", diag = TRUE, upper = TRUE)
eucScaleD = as.matrix(eucScaleD)
dim(eucScaleD)
eucScaleD[1:5, 1:5]

# 6. manhattan distance matrix after scaling by maximum value
manScaleD = dist(scale, method = "manhattan", diag = TRUE, upper = TRUE)
manScaleD = as.matrix(manScaleD)
dim(manScaleD)
manScaleD[1:5, 1:5]

# function of finding out mode
# resource: http://stackoverflow.com/questions/2547402/standard-library-function-in-r-for-finding-the-mode
getMode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# split the cross validation groups
# for this assignment, we use 5-fold cross validation
kfolds = 5
size = nrow(digits)
groupSize = size/kfolds
# folds is a list of vectors, each vector has the length of 1000, which is also the index
# for each group of test data
folds = list(1:groupSize, (groupSize+1):(2*groupSize), (2*groupSize+1):(3*groupSize), 
             (3*groupSize+1):(4*groupSize), (4*groupSize+1):size)

#substrct the labels
trueLabels = digits$label

predictionError =
  function(k, D) {
    # k nearest neighbor model
    # D is the distance matrix
    # the output is the misclassification rate under knn model and D distance matrix
    listOfPredicts = lapply(1:kfolds, function(i) {
      # substract the distance matrix corresponding to each fold of test set and their distances
      # with the rest training set, which has a dimension of 1000*4000
      cv = D[folds[[i]], -folds[[i]]]
      # substract the labels corresponding to the each fold of training set, which has a length of 4000
      cv_label = trueLabels[-folds[[i]]]
      apply(cv, 1, function(row) {
        # row is a row from distance matrix
        row = as.matrix(row)
        # find the correponding labels of the k neighbors
        kNearLabel = cv_label[order(row)][1:k]
        # get the mode of the k lables
        getMode(kNearLabel)
      })
    })
    # we get 5000 predictions for the whole data
    allPredicts = unlist(listOfPredicts)
    # compare predictions with the original data's label and get the number of inconformity
    totalError = sum(digits$label!=allPredicts)
    # define misclasssification rate
    misClassRate = totalError/nrow(digits)
    misClassRate
  }

# create a table to record the misclassfication rate under each combination of 
# k nearest neighbor and distance metric
# D is a list of distance matrix
D = list(eucD, manD, eucStandD, manStandD, eucScaleD, manScaleD)
names(D) = c("eucD", "manD", "eucStandD", "manStandD", "eucScaleD", "manScaleD")
upperK = 30
# errors is a list of 4 vectors, each vector records the errors under a certain distance matrix and 
# knn model with k = 1:30
errors = lapply(D, function(d){
  temp = lapply(1:upperK, function(k){
    predictionError(k, d)
  })
  unlist(temp)
})
# combine this error vectors into a data frame
errorTable = as.data.frame(do.call("cbind", errors))
colnames(errorTable) = names(D)
errorTable
# find the smallest error rate and corresponding model combination
which(errorTable == min(errorTable), arr.ind = TRUE)


# Draw a plot showing the overall cross-validation misclassification rate 
# versus k and the distance metrics
errorTable$k = 1:upperK
library(ggplot2)
ggplot(errorTable, aes(x = k)) + 
  geom_line(aes(y = eucD, colour = "eucD")) + 
  geom_line(aes(y = manD, colour = "manD")) +
  geom_line(aes(y = eucStandD, colour = "eucStandD")) +
  geom_line(aes(y = manStandD, colour = "manStandD")) +
  geom_line(aes(y = eucScaleD, colour = "eucScaleD")) +
  geom_line(aes(y = manScaleD, colour = "manScaleD")) +
  scale_x_continuous(breaks = seq(1, 30, by = 1)) +
  xlab("Number of nearest neighbors") +
  ylab("Average misclassification rate") +
  ggtitle("Average misclassification rate\n under different models")

# Calculate the confusion matrix for the training set using the chosen value of k and metric
# use model k=4 and eucD matrix
predictions =
  function(k, D) {
    # k nearest neighbor model
    # D is the distance matrix
    # the output is the 5000 predicts under knn model and D distance matrix
    listOfPredicts = lapply(1:nrow(D), function(i) {
      # leave one data out, use the rest 4999 as traing set
      row = D[i, -i]
      # substract the labels corresponding to the training set, which has a length of 4999
      row_label = trueLabels[-i]
      row = as.matrix(row)
      # find the correponding labels of the k neighbors
      kNearLabel = row_label[order(row)][1:k]
      # get the mode of the k lables
      getMode(kNearLabel)
    })
    # we get 5000 predictions for the whole data
    unlist(listOfPredicts)    
  }
predicts = predictions(4, eucD)
trueAndPredict = as.data.frame(cbind(trueLabels, predicts))
confusionMatrix = table(trueAndPredict)
confusionMatrix

# Which digits were generally classified best? worst?
accuracy = lapply(1:10, function(i) {
  confusionMatrix[i,i]/sum(confusionMatrix[i,])
})
accuracy = unlist(accuracy)
uniqueLabels = 0:9
accuracyTable = as.data.frame(cbind(uniqueLabels, accuracy))
accuracyTable[order(accuracyTable$accuracy),]

# Which digits were generally confused with others?
confusionMatrix
confusion = lapply(1:10, function(i) {
  1-(confusionMatrix[i,i]/sum(confusionMatrix[,i]))
})
confusion = unlist(confusion)
uniqueLabels = 0:9
confusionTable = as.data.frame(cbind(uniqueLabels, confusion))
confusionTable[order(confusionTable$confusion),]

# Show some of the digits that were mis-classified that were difficult for a human to classify. 
# Suggest why these were misclassified.

# function of drawing an image
getImage =
  function(vals)
  {
    matrix(as.integer(vals), 28, 28, byrow = TRUE)
  }

draw = function(vals, colors = rgb((255:0)/255, (255:0)/255, (255:0)/255), ...)
{
  if(!is.matrix(vals))
    vals = getImage(vals)
  
  m = t(vals)  # transpose the image
  m = m[,nrow(m):1]  # turn up-side-down
  
  image(m, col = colors, ..., xaxt = "n", yaxt = "n")
}


# wrongly predicted digits
wrongPredict = digits[trueAndPredict$trueLabels != trueAndPredict$predicts,]
# how many digits are predicted wrongly
numOfWrong = nrow(wrongPredict)

# look at some mis-classified digits
par(mfrow = c(6,6), mar = c(0,0,1,1) + 0.1)
# digit 2 mis-classified as 1
twoAsOne = digits[trueAndPredict$trueLabels == 2 & trueAndPredict$predicts == 1,][1:12,]
lapply(1:nrow(twoAsOne), function(i) {
  draw(twoAsOne[i,-1])
  return (twoAsOne$label[i])
})
# digit 4 mis-classified as 9
fourAsNine = digits[trueAndPredict$trueLabels == 4 & trueAndPredict$predicts == 9,][1:12,]
lapply(1:nrow(fourAsNine), function(i) {
  draw(fourAsNine[i,-1])
  return (fourAsNine$label[i])
})
# digit 7 mis-classified as 1
sevenAsOne = digits[trueAndPredict$trueLabels == 7 & trueAndPredict$predicts == 1,][1:12,]
lapply(1:nrow(sevenAsOne), function(i) {
  draw(sevenAsOne[i,-1])
  return (sevenAsOne$label[i])
})

# look at one misclassification 
par(mfrow = c(1,1), mar = c(0,0,1,1) + 0.1)
sevenAsOne[2,]
draw(sevenAsOne[2,-1])
rownames(sevenAsOne[2,])

# based on our model, the 4 nearest neighbors of this misclassifications are:
neighbors =
  function(k, D, rowIndex) {
    # k nearest neighbor model
    # D is the distance matrix
    # rowIndex is the index of the digit which we want to find the k nearest neighbors for    
    # the output is the k nearest neighbors 
    row = D[rowIndex,]
    row = as.matrix(row)
    # find the k neighbors, use 2: k+1 to exclude itself as a neighbor
    kNearNeighbor = order(row)[2:(k+1)]
    # get the mode of the k lables
    digits[kNearNeighbor,]
  }

nb = neighbors(4, eucD, '2065')
nb$label
# draw the four neighbors
par(mfrow = c(1,4), mar = c(0,0,1,1) + 0.1)
lapply(1:nrow(nb), function(i) {
  draw(nb[i,-1])
  return(nb$label[i])
})
# draw some true "7"s
par(mfrow = c(4,6), mar = c(0,0,1,1) + 0.1)
someSevens = digits[digits$label == 7,][1:24,]
lapply(1:nrow(someSevens), function(i) {
  draw(someSevens[i,-1])
  return(someSevens$label[i])
})


###############################################################################################
# Distance to Average & Cross-Validation

avgDigits = 
  function(df) {
    # df is the input data frame
    # this function outputs a data frame with 10 rows and 785 columns, each row starts with a digit and 
    # followed by its average 784 pixels
    numbers = 0:9
    ll = lapply(numbers, function(number) {
      sub = subset(df, df$label == number)
      colMeans(sub)
    })
    mm = do.call("rbind", ll)
    as.data.frame(mm)
  }


folds = list(1:groupSize, (groupSize+1):(2*groupSize), (2*groupSize+1):(3*groupSize), 
             (3*groupSize+1):(4*groupSize), (4*groupSize+1):size)


predictionError.avgDist =
  function(df, method) {
    listOfPredicts = lapply(1:kfolds, function(i) {
      test = df[folds[[i]],]
      train = df[-folds[[i]],]
      avg = avgDigits(train)
      predictionForTest =
        lapply(1:nrow(test), function(rowNum) {
          avgDist = dist(rbind(test[rowNum,], avg), method, upper = TRUE, diag = TRUE)
          avgDist = as.matrix(avgDist)
          numbers = 0:9
          numbers[order(avgDist[1,-1])][1]
        })
      unlist(predictionForTest)
    })
    # we get 5000 predictions for the whole data
    allPredicts = unlist(listOfPredicts)
    # compare predictions with the original data's label and get the number of inconformity
    totalError = sum(df$label!=allPredicts)
    # define misclasssification rate
    misClassRate = totalError/nrow(df)
    misClassRate
  }

predictionError.avgDist(digits, "euclidean")
predictionError.avgDist(digits, "manhattan")


