setwd("C:/Users/Ming/Dropbox/statistical learning project/Numerical Method/Code/dev")
source("nqpSolver.R")
source("uilt.R")

data = read.csv("sp500compdata_2013-2015.csv")

data$Date=as.Date(data$Date, format ="%m/%d/%Y")

row.has.na <- apply(data, 1, function(x){any(is.na(x))})

data.filtered <- data[!row.has.na,]

sp500 = read.csv("sp 500 index_2013_2015_yahoo.csv")

sp500$Date=as.Date(sp500$Date, format ="%m/%d/%Y")

sp500 = sp500[c("Date","Adjusted.Close")]

new_data = merge(data.filtered,sp500,by = c("Date"))

new_data = new_data[-1]

l = dim(new_data)[1]

rates = (new_data[2:l,]-new_data[1:l-1,])/new_data[1:l-1,]

require(caret)

n_fold = 10

flds <- createFolds(rates, k = n_fold,list = TRUE, returnTrain = FALSE)

lambda_grid = c(0,0.01,0.1,1,10,100,1000)

alpha_grid = seq(0,1,0.1)

# para_grid = apply(expand.grid(lambda_grid,alpha_grid), 1, function(x) c(x[1],x[2]))

n_total = length(lambda_grid)*length(alpha_grid)
counter = 1
  
mse_table = c()
for(i in 1:length(lambda_grid)){
  
  lambda <- lambda_grid[i]
  
  for(j in 1: length(alpha_grid)){
    
    alpha <- alpha_grid[j]
    mses = c()
    for (k in seq(1,n_fold,1)){
      
      testing_ind <- flds[[k]]
      training_data <- rates[-testing_ind,]
      x <- as.matrix(cbind(rep(1,dim(training_data)[1]),training_data[,1:(dim(training_data)[2]-1)]))
      y <- as.matrix(training_data[,dim(training_data)[2]])
      testing_data <- rates[testing_ind,]
      testing_x <- as.matrix(cbind(rep(1,dim(testing_data)[1]),testing_data[,1:(dim(testing_data)[2]-1)]))
      testing_y <- as.matrix(testing_data[,dim(testing_data)[2]])
    
      
      I <- diag(1,ncol(x))
      A <- 2*(t(x)%*%x+lambda*(1-alpha)*I)
      b <- lambda*alpha*rep(1,length(ncol(x))) - 2*t(y)%*%x
      sol <- as.matrix(munqp(A,b))
      print(sum(sol>1e-2))
      
      
      testing_y_hat <- predict(testing_x,sol)
      mse = calculateMSE(testing_y,testing_y_hat)
      mses = c(mses,mse)
    
    }
    mse_table=rbind(mse_table,c(lambda,alpha,mean(mses)))
    
    
    print(counter/n_total)
    counter=counter+1
    
  }
  
}
print(mse_table)

mse_table= as.data.frame(mse_table)

colnames(mse_table)=c("lambda","alpha","avg_mse")


ggplot(data = mse_table,aes(x=alpha,y=avg_mse,colour = factor(lambda)))+geom_line()+geom_point()
