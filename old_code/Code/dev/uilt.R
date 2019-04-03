predict <- function(x, beta){
  return(-x %*% beta)
  
}

calculateMSE <- function(y,y1){
  return(mean((y-y1)^2))
}