munqp <- function(Amat, bvec, err = 1e-5){
  
        A_plus <- Amat
        A_plus[A_plus<0] <- 0
        
        A_minus <- Amat
        A_minus[A_minus>0] <- 0
        A_minus <- abs(A_minus)
        
        v = rep(1,length(bvec))
        
        updateFactor <- rep(0,length(bvec))

        
        while (sum((t(updateFactor)*v-v)^2) > err){
                dFa <- A_plus %*% v
                
                dFb <- bvec
                
                dFc <- A_minus %*% v
                
                for(i in 1:length(bvec)){
                        updateFactor[i] <- (-dFb[i]+sqrt(dFb[i]^2+4*dFa[i]*dFc[i]))/(2*dFa[i])
                }
                if(sum(is.na(updateFactor))==0){
                        v <- (updateFactor)*v
                }else{
                        break
                }
        }
        
        return(v)
        
}