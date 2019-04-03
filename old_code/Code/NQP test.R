source("nqpSolver.R")
A = matrix(c(5,-2,-1,-2,4,3,-1,3,5),byrow = TRUE,nrow = 3)
A

CA = matrix(c(0,0,0,0,0,0,0,0,0),byrow = TRUE,nrow = 3)
CA

b = t((c(2,-35,-47)))
b

bvec = t(c(0,0,0))

require(quadprog)

sol <- solve.QP(A,-b,CA,bvec)
sol

sol1 <- munqp(A,b)

sol1


## Assume we want to minimize: -(0 5 0) %*% b + 1/2 b^T b
## under the constraints: A^T b >= b0
## with b0 = (-8,2,0)^T
## and (-4 2 0)
## A = (-3 1 -2)
## ( 0 0 1)
## we can use solve.QP as follows:
##
Dmat <- matrix(0,3,3)
diag(Dmat) <- 1
dvec <- t(-c(0,5,0))
Amat <- matrix(c(0,0,0,0,0,0,0,0,0),3,3)
bvec <- t(c(0,0,0))
solve.QP(Dmat,-dvec,Amat,bvec=bvec)

sol1 <- munqp(Dmat,dvec)


sol1


install.packages("devtools")


library(devtools)

install.packages("Quandl")

library(Quandl)

Quandl.api_key("SuYpYx5ME39fEvwsgHPp")

csi300 = Quandl("YAHOO/SS_000300", type = "raw",start_date="2005-10-01", end_date="2011-11-30")

csi300 = csi300$`Adjusted Close`

l = length(csi300)

rcsi300 = (csi300[2:l]/csi300[1:(l-1)] - 1)





