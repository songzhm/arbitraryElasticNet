A <- matrix(c(5,-2,-1,-2,4,3,-1,3,5),byrow = TRUE,nrow = 3)
b <- c(2,-35,-47)

A_plus <- A
A_plus[A_plus<0] <- 0
A_plus

A_minus <- A
A_minus[A_minus>0] <- 0
A_minus <- abs(A_minus)

v <- c(1,1,1)

dFa <- A_plus %*% v

dFb <- as.matrix(b,nrow = length(b))

dFc <- A_minus %*% v

dFa
dFb
dFc

updateFactor <- c()

for(i in 1:length(b)){
        updateFactor <- cbind(updateFactor,(-dFb[i]+sqrt(dFb[i]^2+4*dFa[i]*dFc[i]))/(2*dFa[i]))
}

sum((t(updateFactor)*v-v)^2)
v <- t(updateFactor)*v
v
