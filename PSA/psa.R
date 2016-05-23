############################### STA206 Final Project #############################
# read data into R called "prostate"
prostate = prostate[,-1]
names = c("psa", "cancerVolume", "weight", "age", 
          "benignPH", "seminalVI", "capsularP", "gleasonScore")
names(prostate) = names
# check if there is missing value
sum(is.na(prostate)) # no missing value

############################### data exploration #################################
# summary statistics for each variable 

# exploration
sapply(prostate, class) 
# seminalVI and gleasonScore care categorical, get their levels
quantiVar = c("psa", "cancerVolume", "weight", "age", "benignPH", "capsularP")
cateVar = c("seminalVI", "gleasonScore")
sapply(cateVar, function(x) unique(prostate[,which(names == x)]))
# change the class of categorical data into factor
prostate$seminalVI = factor(prostate$seminalVI)
prostate$gleasonScore = factor(prostate$gleasonScore)
# draw pie charts for categorical data
par(mfrow = c(1,2), mar=c(1,1,1,1))
sapply(cateVar, function(x) pie(table(prostate[,which(names == x)]), col = rainbow(3), 
                                main = paste("Pie chart of", x)))

# draw boxplots for categorical data
par(mfrow = c(1,2), mar = c(2,2,2,2))
sapply(cateVar, function(x) boxplot(prostate$psa~prostate[,which(names == x)], 
                                    main = paste("Boxplot of", x),
                                    xlab = x, ylab = "psa", col = rainbow(3)))
# for quantitative variables, get the distributions
par(mfrow=c(2,3))
sapply(quantiVar, function(x) hist(prostate[,which(names == x)], main = x))

# pairwise correlation between quantitative variables
pairs(prostate[,which(names %in% quantiVar)])
cor = cor(prostate[,which(names %in% quantiVar)]) 
library(corrplot)
par(mfrow=c(1,1), mar=c(1,1,1,1))
corrplot(cor, method = "circle", pch = 1, tl.cex=0.75)
corrplot(cor, method = "number", pch = 1, tl.cex=0.75)

# multicolinearity investigation 
r.XX = cor(prostate[,which(names %in% quantiVar)][,-1])
diag(solve(r.XX)) # there is no much multicolinearity in these variables

# bivariate regression
fit.cancerVol = lm(psa~cancerVolume, data = prostate)
summary(fit.cancerVol)

fit.weight = lm(psa~weight, data = prostate)
summary(fit.weight)

fit.age = lm(psa~age, data = prostate)
summary(fit.age)

fit.benignPH = lm(psa~benignPH, data = prostate)
summary(fit.benignPH)

fit.seminalVI = lm(psa~seminalVI, data = prostate)
summary(fit.seminalVI)

fit.capsularP = lm(psa~capsularP, data = prostate)
summary(fit.capsularP)

fit.gleasonScore = lm(psa~gleasonScore, data = prostate)
summary(fit.gleasonScore)

# transformation
psa_star = log((prostate$psa)) # psa
cancerVolume_star = log((prostate$cancerVolume)) # cancer volume
weight_star = log((prostate$weight)) # weight
# see the histograms of transformed variables
par(mfrow=c(3,2), mar=c(2,2,1,2))
hist(prostate$psa, main = "psa")
hist(psa_star, main = "log psa +0.35")
hist(prostate$weight, main = "weight")
hist(weight_star, main="log weight")
hist(prostate$cancerVolume, main = "cancer volume")
hist(cancerVolume_star, main = "log cancer volume+0.75")
# data set with psa transformed: prostate_star1
prostate_star1 = prostate
prostate_star1$psa = psa_star
# data set with psa, cancerVolume transformed: prostate_star2
prostate_star2 = prostate
prostate_star2$psa = psa_star
prostate_star2$cancerVolume = cancerVolume_star
# data set with psa, weight, cancerVolume transformed: prostate_star2
prostate_star3 = prostate
prostate_star3$psa = psa_star
prostate_star3$weight = weight_star
prostate_star3$cancerVolume = cancerVolume_star
# fit data with original data set, prostate_star1 and prostate_star2 respectively
tmp1=lm(psa~., data=prostate)
tmp2=lm(psa~., data=prostate_star1)
tmp3=lm(psa~., data=prostate_star2)
tmp4=lm(psa~., data=prostate_star3)
# compare residual plots and QQ plots
par(mfrow=c(4,2), mar=c(2,2,2,2))
plot(tmp1, which=1, caption='', main="Residuals VS fitted for original data")
plot(tmp1, which=2, caption='', main="Q-Q plot for original data")
plot(tmp2, which=1, caption='', main="Residuals VS fitted for data with log(psa)")
plot(tmp2, which=2, caption='', main="Q-Q plot for data with log(psa)")
plot(tmp3, which=1, caption='', 
     main="Residuals VS fitted for data with log(psa), log(cancerVolume)")
plot(tmp3, which=2, caption='', 
     main="Q-Q plot for for data with log(psa), log(cancerVolume)")
plot(tmp4, which=1, caption='', 
     main="Residuals VS fitted for data with log(psa), log(weight), log(cancerVolume)")
plot(tmp4, which=2, caption='', 
     main="Q-Q plot for for data with log(psa), log(weight), log(cancerVolume)")
# we decide using transformed psa and transformed cancer volume

############################ model 1 with first order only ###########################
# split data into training set and testing set
set.seed(2062015)
# randomly take 72 out of 97 into training set and put the rest 25 into testing group
index = sample(1: 97, 72, replace = FALSE)
prostate.train = prostate_star2[index,]
prostate.test = prostate_star2[index,]
# full model with all first order terms
fit1 = lm(psa ~., data = prostate.train)
summary(fit1)
anova(fit1)
par(mfrow = c(1,2))
plot(fit1, which = 1) # nonlinear
plot(fit1, which = 2) # left skewed
mse.fit1 =anova(fit1)['Residuals',3]
fit0 = lm(psa~1, data = prostate.train) ##fit the model with only intercept

# best subset selection
# best subset model selection
library(leaps)
sub_set = regsubsets(psa~., data = prostate.train, nbest = 2, nvmax = 8, method = "exhaustive")
sum_sub = summary(sub_set)

# number of coefficients in each model: p
n = 72
p.m = as.integer(as.numeric(rownames(sum_sub$which))+1)
sse = sum_sub$rss
aic = n*log(sse/n)+2*p.m
bic = n*log(sse/n)+log(n)*p.m
res_sub = cbind(sum_sub$which, sse, sum_sub$rsq, sum_sub$adjr2, sum_sub$cp, aic, bic)
sse1 = sum(fit0$residuals^2)
p = 1
c1 = sse1/mse.fit1-(n-2*p)
aic1 = n*log(sse1/n)+2*p
bic1 = n*log(sse1/n)+log(n)*p
none = c(1, rep(0,8), sse1, 0, 0, c1, bic1, aic1)
res_sub = rbind(none,res_sub) ##combine the results with other models
colnames(res_sub) = c(colnames(sum_sub$which),"sse", "R^2", "R^2_a", "Cp", "aic", "bic")
res_sub

library(xlsx)
write.xlsx(res_sub, "/Users/PullingCarrot/Desktop/206/modelSelect1.xlsx")
# according to subset selection method, the best model is as follows with AIC = -52.31
model1 = lm(psa~ cancerVolume+weight+seminalVI+gleasonScore, data=prostate.train)
summary(model1)
par(mfrow=c(1,2))
plot(model1, which=1) 
plot(model1, which=2) # seems a good fit

# stepwise model selection
step.f1=stepAIC(fit0, scope=list(upper=fit1, lower=~1), direction = "both", k=2)
# according to stepwise selection, the preferred model is as follows
# however, this model is suboptimal with AIC=-50.48, which is larger than AIC of model1.
model1.step = lm(psa ~ cancerVolume + gleasonScore + weight + seminalVI + benignPH,
                 data=prostate.train)

####################### model 2 with first order and second order #######################
# investigate quantitative variables' interaction 
par(mfrow=c(3,4))
i=2
while (i < 7) {
  j = i+1
  while (j < 7){
    plot(prostate.train[, which(names %in% quantiVar[i])]*
           prostate.train[, which(names %in% quantiVar[j])],
         fit1$residuals, main = paste(quantiVar[i],"and", quantiVar[j]))
    j=j+1
  }
  i=i+1  
}

# investigate interation between quantitative and categorical variables
library(ggplot2)
ggplot(prostate.train, aes(x=age, y=psa)) +
  geom_point(aes(color=gleasonScore)) +
  stat_smooth(aes(color=gleasonScore),method="lm", se=FALSE)

p1 = 
  ggplot(prostate.train, aes(x=cancerVolume, y=psa)) +
  geom_point(aes(color=gleasonScore)) +
  stat_smooth(aes(color=gleasonScore),method="lm", se=FALSE) 

p2 = 
  ggplot(prostate.train, aes(x=weight, y=psa)) +
  geom_point(aes(color=gleasonScore)) +
  stat_smooth(aes(color=gleasonScore),method="lm", se=FALSE) 

p3 = 
  ggplot(prostate.train, aes(x=age, y=psa)) +
  geom_point(aes(color=gleasonScore)) +
  stat_smooth(aes(color=gleasonScore),method="lm", se=FALSE) 

p4 = 
  ggplot(prostate.train, aes(x=benignPH, y=psa)) +
  geom_point(aes(color=gleasonScore)) +
  stat_smooth(aes(color=gleasonScore),method="lm", se=FALSE) 

p5 = 
  ggplot(prostate.train, aes(x=capsularP, y=psa)) +
  geom_point(aes(color=gleasonScore)) +
  stat_smooth(aes(color=gleasonScore),method="lm", se=FALSE) 
library(gridExtra)
grid.arrange(p1,p2,p3,p4,p5,ncol=3)

g1 = 
  ggplot(prostate.train, aes(x=cancerVolume, y=psa)) +
  geom_point(aes(color=seminalVI)) +
  stat_smooth(aes(color=seminalVI),method="lm", se=FALSE) 
g2 = 
  ggplot(prostate.train, aes(x=weight, y=psa)) +
  geom_point(aes(color=seminalVI)) +
  stat_smooth(aes(color=seminalVI),method="lm", se=FALSE) 
g3 = 
  ggplot(prostate.train, aes(x=age, y=psa)) +
  geom_point(aes(color=seminalVI)) +
  stat_smooth(aes(color=seminalVI),method="lm", se=FALSE) 
g4 = 
  ggplot(prostate.train, aes(x=benignPH, y=psa)) +
  geom_point(aes(color=seminalVI)) +
  stat_smooth(aes(color=seminalVI),method="lm", se=FALSE) 
g5 = 
  ggplot(prostate.train, aes(x=capsularP, y=psa)) +
  geom_point(aes(color=seminalVI)) +
  stat_smooth(aes(color=seminalVI),method="lm", se=FALSE) 
grid.arrange(g1,g2,g3,g4,g5,ncol=3)
# we are going to do hypothesis tests to see whether the interaction terms 
# should be included in
inter.model = 
  lm(psa~. +cancerVolume:gleasonScore+age:gleasonScore+weight:gleasonScore+
            benignPH:gleasonScore+weight:seminalVI+age:seminalVI+
            benignPH:seminalVI+capsularP:seminalVI, 
     data=prostate.train)
summary(inter.model)
anova(inter.model)
ssr.inter=0.198+0.009+3.277+0.493+0.579+0.197+0.102+0.034
f.stat = (ssr.inter/8)/(24.07/51)
pValue = pf(f.stat,8,51) # p value is 0.733, so that we can drop the interactin terms

# add all the second order into the full model
fit2 = 
  lm(psa~.^2 + I(cancerVolume^2) + I(weight^2) + I(age^2) + I(benignPH^2) + I(capsularP^2), 
     data = prostate.train)
summary(fit2)
par(mfrow=c(1,2))
plot(fit2, which=1)
plot(fit2, which=2)
mse.fit2 =anova(fit2)['Residuals',3]

# best subset selection
sub_set2 = 
  regsubsets(psa~.^2 + I(cancerVolume^2) + I(weight^2) + I(age^2) + I(benignPH^2) + 
               I(capsularP^2), 
             data = prostate.train, nbest = 1, nvmax = 41, method = "exhaustive")
sum_sub2 = summary(sub_set2)

n = 77
p.m2 = as.integer(as.numeric(rownames(sum_sub2$which))+1)
sse2 = sum_sub2$rss
aic2 = n*log(sse2/n)+2*p.m2
bic2 = n*log(sse2/n)+log(n)*p.m2
res_sub2 = cbind(sum_sub2$which, sse2, sum_sub2$rsq, sum_sub2$adjr2, sum_sub2$cp, aic2, bic2)
c2 = sse1/mse.fit2-(n-2*p)
none2 = c(1, rep(0,40), sse1, 0, 0, c2, bic1, aic1)
res_sub2 = rbind(none2,res_sub2) ##combine the results with other models
colnames(res_sub2) = c(colnames(sum_sub2$which),"sse", "R^2", "R^2_a", "Cp", "aic", "bic")
res_sub2
write.xlsx(res_sub2, "/Users/PullingCarrot/Desktop/206/modelSelect2.xlsx")

# AIC=-90.57
model2 = lm(psa~cancerVolume + age + benignPH + seminalVI + gleasonScore + weight + 
              capsularP + I(weight^2) + I(cancerVolume^2) + I(capsularP^2) +
              weight:benignPH + weight:capsularP + cancerVolume:gleasonScore + 
              seminalVI:capsularP, 
              data=prostate.train)
summary(model2)
par(mfrow=c(1,2), mar=c(2,2,2,2))
plot(model2, which=1)
plot(model2, which=2)

# stepwise model selection: AIC=-56.38
stepAIC(fit0, scope=list(upper=fit2, lower=~1), direction = "both", k=2)
model2.step = lm(formula = psa ~ cancerVolume + gleasonScore + weight + seminalVI + 
                 I(capsularP^2) + I(cancerVolume^2) + weight:seminalVI + gleasonScore:weight, 
                 data = prostate.train)
par(mfrow=c(1,2), mar=c(2,2,2,2))
plot(model2.step, which=1)
plot(model2.step, which=2)

################################# model validation ###########################################
# internal validation: Cp and PRESSp
sse.best1 = anova(bestmodel1)["Residuals",2]
sse.best2 = anova(bestmodel2)["Residuals",2]
mse.best1 = anova(bestmodel1)["Residuals",3]
mse.best2 = anova(bestmodel2)["Residuals",3]
press.best1 = sum(bestmodel1$residuals^2/(1-influence(bestmodel1)$hat)^2)
press.best2 = sum(bestmodel2$residuals^2/(1-influence(bestmodel2)$hat)^2)

# cross validation
newdata = prostate.test[,-1]
pred.best1 = predict.lm(bestmodel1, newdata)
mspe.best1 = mean((pred.best1-prostate.test[,1])^2)
mspe.best1
pred.best2 = predict.lm(bestmodel2, newdata)
mspe.best2 = mean((pred.best2-prostate.test[,1])^2)
mspe.best2

################################# final model ###########################################
# besed on PRESSp and mspe, bestmodel2 is better, use it as final model
# fit the final model in the entire data set
finalmodel = lm(psa~cancerVolume + age + benignPH + seminalVI + gleasonScore + weight + 
                  capsularP + I(weight^2) + I(cancerVolume^2) + I(capsularP^2) +
                  weight:benignPH + weight:capsularP + cancerVolume:gleasonScore + 
                  seminalVI:capsularP, data=prostate_star2)
summary(finalmodel)
anova(finalmodel)

################################# model diagnostic ###########################################
# residual plot and QQ plot
par(mfrow=c(1,2), mar=c(2,2,2,2))
plot(finalmodel, which=1) # constancy in error variance 
plot(finalmodel, which=2) # heavy tails may due to outliers

# outliers investigation
# identify any outlying response variable observations
res = residuals(finalmodel)
nn = nrow(prostate_star2)
pp = 15
hh = influence(finalmodel)$hat
d.res.std = studres(finalmodel) # studendized deleted residuals
sort(abs(d.res.std), decreasing = TRUE)
idx.Y = as.vector(which(abs(d.res.std)>=qt(1-0.1/(2*nn), nn-pp-1)))# bonferronis threshold
idx.Y # case 32 is outlier in response value

# identify any outlying independent observations
idx.X = as.vector(which(hh>2*pp/nn))
idx.X # outliers: 32 41 59 69 70 76 82 89 91 92 97
par(mfrow=c(1,1), mar=c(4,4,4,4))
plot(res, hh, xlab = "leverage", ylab="residuals", main="Residuals vs. leverage plot")

# identify influential cases
plot(finalmodel, which=4) # case 32 is outlying severely
mse.final = anova(finalmodel)["Residuals", 3]
cook.d = res^2*hh/(pp*mse.final*(1-hh)^2)
cook.max = cook.d[which(cook.d==max(cook.d))]
pf(cook.max,pp,nn-pp)
idx = c(idx.X,idx.Y)
cook.d[idx]
pf(cook.d[idx],pp,nn-pp)
# case 32 is an influential case. Drop it and fit the final model again
finalmodel2 = lm(psa~cancerVolume + age + benignPH + seminalVI + gleasonScore + weight + 
                  capsularP + I(weight^2) + I(cancerVolume^2) + I(capsularP^2) +
                  weight:benignPH + weight:capsularP + cancerVolume:gleasonScore + 
                  seminalVI:capsularP, data=prostate_star2[-32,])
summary(finalmodel2)
anova(finalmodel2)
par(mfrow=c(1,3), mar=c(2,2,2,2))
plot(finalmodel2, which=1) # constancy in error variance 
plot(finalmodel2, which=2) # light tails 
plot(finalmodel2, which=4) # now the boxcox is much normal