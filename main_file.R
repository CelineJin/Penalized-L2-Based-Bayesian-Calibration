setwd("/Users/celinejin/Google Drive/My Drive/Research/MSEN655/Code2D")
install.packages("numDeriv")
install.packages("sensitivity")
install.packages("kernlab")
install.packages("ggplot2")
install.packages("gdata")

library(lattice)
library(numDeriv)
library(sensitivity)
library(kernlab)
library(ggplot2)
library(gdata)

source("KLR.R")
source("cv.KLR.R")
source("cv.gausspr.R")

start.time <- Sys.time()

# Tuning parameter dimension = 2
dtune <- 2 #fixed for 2-dim parameters calibration

## Need user's intervention---change according to input scenarios
sname <- 30
suffix <- "_symsmall"

filenames_sim <- paste("Input2D/track54_NiNb_KGT_simdata_",sname,suffix,".csv",sep="")
simdata.df <- read.csv(filenames_sim, header=FALSE)
simdata <- simdata.df
np <- dim(simdata)[1]/6

filenames_lab <- paste("Input2D/track54_NiNb_KGT_labdata_",sname,suffix,".csv",sep="")
labdata.df <- read.csv(filenames_lab, header=FALSE)
labdata <- labdata.df

##### standardizing input variables #####
# lab data #
xp <- labdata[,1:2]
xp[,2] <- xp[,2] + 1 - min(xp[,2])
xp[,2] <- log(xp[,2])
range.diff.xp <- apply(xp, 2, function(x) diff(range(x)))
min.xp <- apply(xp, 2, min)
xp <- apply(xp, 2, FUN = function(x) (x-min(x))/diff(range(x)))
yp <- labdata[,3]


Xp <- xp 
yp <- yp

np <- nrow(Xp)
d <- ncol(Xp)

# simulation data #
xs <- simdata[,1:(dtune+2)]
xs[,2] <- xs[,2]+1-min(xs[,2])      # xs is LHD after using log transformation
xs[,2] <- log(xs[,2])
min.xs <- apply(xs, 2, min)
max.xs <- apply(xs, 2, max)
range.diff.xs <- apply(xs, 2, function(x) diff(range(x)))
xs3.range.diff <- range.diff.xs[1+dtune]
xs4.range.diff <- range.diff.xs[2+dtune]
for(i in 1:(dtune+2)) xs[,i] <- (xs[,i] - min.xs[i])/range.diff.xs[i]
ys <- simdata[,dtune+3]
Xs1 <- xs[,1:2]
Xs2 <- xs[,(1+dtune):(2+dtune)]


Xs <- cbind(Xs1, Xs2)
ns <- nrow(Xs)
q <- ncol(Xs)-d
ys <- as.factor(ys)
## set parameters
K = 10
lambda = seq(0.001,0.1,0.005)
kernel = c("matern","exponential")[1]
nu = 2.5
power = 1.95
rho = seq(0.05,0.5,0.05)
sigma = seq(100,20,-1)

## TRAINING STARTS
#### Fit lab data with KLR ####
cv.out <- cv.KLR(Xp, yp, K = K, lambda = lambda, kernel = kernel, nu = nu, power = power, rho = rho)

x.test <- randtoolbox::sobol(20^d, dim = d)  # for integration
if(d == 1) x.test <- matrix(x.test, ncol=1)

cat("Running kernel logistic regression model.\n")
etahat <- KLR(Xp, yp, x.test, lambda = cv.out$lambda, kernel = kernel, nu = nu, power = power, rho = cv.out$rho)

N1 <- sum(etahat>0.5)
N2 <- sum(etahat<=0.5)

#### Fit sim data with GP ####
cat("Running Gaussian process regression model.\n")
sigma.hat <- cv.gausspr(Xs, ys, K=K, sigma=sigma)
gp.fit <- gausspr(Xs, ys, 
                  kpar=list(sigma=sigma.hat[[1]]), 
                  type="classification", scaled=FALSE, cross = 0, fit=FALSE)


L2_fun <- function(theta) {
  #lambda0 <- 1
  x.new <- cbind(x.test, matrix(theta, ncol = q, nrow = nrow(x.test), byrow = TRUE))
  sqrt(mean((etahat - predict(gp.fit, x.new, type="probabilities")[,2])^2))+sqrt(sum(pmin(predict(gp.fit, x.new, type="probabilities")[,2]-etahat,0)^2)/N1)+sqrt(sum(pmax(predict(gp.fit, x.new, type="probabilities")[,2]-etahat,0)^2)/N2)
}

ini.val <- randtoolbox::sobol(3^q, q)

cat("Optimizing parameters.\n")
if(q==1) ini.val <- matrix(ini.val,ncol=1)
opt.out <- vector("list", 3^q)
for(i in 1:3^q) opt.out[[i]] <- optim(ini.val[i,], L2_fun, lower = rep(0,q), upper = rep(1,q), method = "L-BFGS-B")
opt.val <- sapply(opt.out, function(x) x$value)
opt.sol <- sapply(opt.out, function(x) x$par)
if(q > 1) opt.sol <- t(opt.sol)
# opt.sol <- opt.sol * xs3.range.diff + xs3.min

out <- cbind(opt.sol, opt.val)
colnames(out) <- c(paste0("par.",1:q), "l2 dist")
out <- out[sort.int(out[,"l2 dist"], index.return = TRUE)$ix,]
out <- unique(round(out, 5))

#####      scale back to original range     #####
cat("scaling back to the original region.\n")

out[, 1:q] <- t((t(out[, 1:q]) * (max.xs[(d+1):(d+q)] - min.xs[(d+1):(d+q)]) + min.xs[(d+1):(d+q)]))
out


## calculate L2 and objective 
obj_val <- c()
L2_val <- c()
cont_neg <- c()
cont_pos <- c()
for (i in 1:length(out)/(1+dtune)) {
  theta <- opt.sol[i,]
  x.new <- cbind(x.test, matrix(theta, ncol = q, nrow = nrow(x.test), byrow = TRUE))
  obj_val[i] <- eval(obj_cal(x.new))
  L2_val[i] <- eval(L2_cal(x.new))
  cont_neg[i] <- eval(neg_cal(x.new))
  cont_pos[i] <- eval(pos_cal(x.new))
}

file.out <- paste("out_",sname,suffix,".csv",sep="")
write.csv(out,file.out)

