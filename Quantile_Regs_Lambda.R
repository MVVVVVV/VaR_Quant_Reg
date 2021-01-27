library(rqPen)
library(hqreg)
library(data.table)
library(tibble)

var_quantile = 0.01
max_sum_betas = 1
rolling_windows = c(250,500,1000,1500, "ERW")
start_date = "2007-01-03"
end_date = "2014-12-31"
my_path = "C:\\Users\\MV\\Desktop\\Projet Quant\\"
input_assets_path = "C:\\Users\\MV\\Documents\\GitHub\\VaR_Quant_Reg\\Assets\\"
output_path = "C:\\Users\\MV\\Documents\\GitHub\\VaR_Quant_Reg\\Results_RQ\\"
  
rdt <- read.csv(paste(my_path,"rdt.csv", sep=""), header=TRUE, row.names="Date")
#%%%%%%%%%%%%%

start_index = grep(start_date, row.names(rdt))
end_index = grep(end_date, row.names(rdt))
nbr_prediction = end_index - start_index +1

var_prediction <- data.frame(matrix(ncol = 8, nrow = nbr_prediction))
var_col_names <- c("Date","lasso_bic", "ridge_cv", "elastic_cv", "lasso_cv",
                   "ridge_fix", "elastic_fix", 'lasso_fix')
colnames(var_prediction) <- var_col_names
n = 1

# on crée nos lambdas sur echelle de 10
log_values <- cbind(0.1,1,10,100,1000,10000,100000)
nb_steps = 13
lambdas = seq(0,1,1/nb_steps)*0.01

for (i in 1:length(log_values)){
  temp = seq(0,1,1/(nb_steps))*log_values[i]
  lambdas = append(lambdas, temp[2:length(temp)])
}

# on itere sur nos 4 rolling windows
for (RW in rolling_windows){
  index_RW = 1
  # cette matrice comprendra les tick loss par actifs par lambdas pour chaque modele
  tick_loss_per_lambda_lasso  = matrix(,nrow=30,ncol = length(lambdas))
  tick_loss_per_lambda_ridge = matrix(,nrow=30,ncol = length(lambdas))
  tick_loss_per_lambda_elastic = matrix(,nrow=30,ncol = length(lambdas))
  
  # ces matrices comprendront la moyenne des tick loss des 30 actifs par fenetre par lambda
  tick_loss_per_lambda_lasso_per_RW = matrix(,nrow=length(rolling_windows),ncol = length(lambdas))
  tick_loss_per_lambda_ridge_per_RW = matrix(,nrow=length(rolling_windows),ncol = length(lambdas))
  tick_loss_per_lambda_elastic_per_RW = matrix(,nrow=length(rolling_windows),ncol = length(lambdas))
  
  # les predictions pour chaque actif pour chaque date
  var_prediction_all_asset_lasso_bic = matrix(, nrow=nbr_prediction, ncol = 30)
  var_prediction_all_asset_ridge_cv = matrix(, nrow=nbr_prediction, ncol = 30)
  var_prediction_all_asset_elastic_cv = matrix(, nrow=nbr_prediction, ncol = 30)
  var_prediction_all_asset_lasso_bic = matrix(, nrow=nbr_prediction, ncol = 30)
  var_prediction_all_asset_ridge_fix = matrix(, nrow=nbr_prediction, ncol = 30)
  var_prediction_all_asset_elastic_fix = matrix(, nrow=nbr_prediction, ncol = 30)
  var_prediction_all_asset_lasso_bic = matrix(, nrow=nbr_prediction, ncol = 30)
  asset_index = 1
  for(asset in colnames(rdt)){
    
    var = read.csv(paste(input_assets_path,"models_asset_",(asset_index-1)))
    # les matrices var_predictions comprennent nos prévisions pour cet actif
    # avec en row les dates et en col les differents lambdas
    var_prediction_lasso = matrix(,nrow = nbr_prediction, ncol = length(lambdas))
    var_prediction_ridge = matrix(,nrow = nbr_prediction, ncol = length(lambdas))
    var_prediction_elastic = matrix(,nrow =nbr_prediction, ncol = length(lambdas))
    lasso_bic = matrix(,nrow = nbr_prediction, ncol = 30)
    ridge_cv =matrix(,nrow = nbr_prediction, ncol = 30)
    elastic_cv =matrix(,nrow = nbr_prediction, ncol = 30)
    lasso_cv =matrix(,nrow = nbr_prediction, ncol = 30) 
    ridge_fix =matrix(,nrow = nbr_prediction, ncol = 30)
    elastic_fix =matrix(,nrow = nbr_prediction, ncol = 30)
    lasso_fix =matrix(,nrow = nbr_prediction, ncol = 30)
    # la matrice qui va contenir les predictions pour les differents QR par dates
    var_prediction_asset = matrix(,nrow=nbr_prediction, ncol = 7)
    
    t0 <- Sys.time()
    for (date_row in seq(start_index,end_index, by=1)){
      
      if (RW =="ERW"){
        start_rol_win = 1
      }else{
        start_rol_win = date_row-as.numeric(RW)-1
      }
      end_rol_win = date_row-1
      my_rdt = rdt[start_rol_win:end_rol_win, asset]
      my_var = var[start_rol_win:end_rol_win, asset]
      fit_lasso = hqreg(my_rdt, my_var, method = "quantile", tau = var_quantile, alpha = 1, 
                        lambda = lambdas)
      fit_ridge = hqreg(my_rdt, my_var, method = "quantile", tau = var_quantile, alpha = 0,
                        lambda = lambdas)
      fit_elastic = hqreg(my_rdt, my_var, method = "quantile", tau = var_quantile, alpha = 0.5, 
                          lambda = lambdas)
      
      var_prediction_lasso[date_row,] = predict(fit_lasso, var[date_row,asset],
                                     lambda = lambdas)
      var_prediction_ridge[date_row,] = predict(fit_ridge, var[date_row,asset],
                                     lambda = lambdas)
      var_prediction_elastic[date_row,] = predict(fit_elastic, var[date_row,asset],
                                     lambda = lambdas)
      
     
    }
    t1<-Sys.time()
    print(t1-t0)
    
    tick_loss_per_lamdbda_lasso[asset_index,] =  tick_loss(rdt, var_prediction_lasso, var_quantile)
    tick_loss_per_lambda_ridge[asset_index,] = tick_loss(rdt, var_prediction_ridge, var_quantile)
    tick_loss_per_lambda_elastic[asset_index,] = tick_loss(rdt, var_prediction_elastic, var_quantile)
    
    # on selectionne le meilleure lambda en fct des differents criteres
    index_lasso_bic =which.min(BIC(my_rdt, var_prediction_lasso, var_quantile,
                                  fit_lasso$beta))
    index_ridge_cv =which.min(cross_validation(rdt=my_rdt, var = var_prediction_ridge,
                                               var_quantile = var_quantile))
    index_elastic_cv =which.min(cross_validation(rdt=my_rdt, var = var_prediction_elastic,
                                                 var_quantile = var_quantile))
    index_lasso_cv = which.min(cross_validation(rdt=my_rdt, var = var_prediction_lasso,
                                                var_quantile = var_quantile))
    index_ridge_fix =heuristic(lambda=fit_ridge$lambda,
                         betas=fit_ridge$beta[2:.N,] )
    index_elastic_fix =heuristic(lambda=fit_elastic$lambda,
                           betas=fit_elastic$beta[2:.N,] )
    index_lasso_fix =heuristic(lambda=fit_lasso$lambda,
                         betas=fit_lasso$beta[2:.N,] )
    
    
    # on ajoute l'asset pour chaque modele
    var_prediction_all_asset_lasso_bic[, asset_index] = var_prediction_lasso[,index_lasso_bic]
    var_prediction_all_asset_ridge_cv[, asset_index] = var_prediction_asset[,index_ridge_cv]
    var_prediction_all_asset_elastic_cv[, asset_index] = var_prediction_asset[,index_elastic_cv]
    var_prediction_all_asset_lasso_cv[, asset_index] = var_prediction_asset[,index_lasso_cv]
    var_prediction_all_asset_ridge_fix[, asset_index] = var_prediction_asset[,index_ridge_fix]
    var_prediction_all_asset_elastic_fix[, asset_index] = var_prediction_asset[,index_elastic_fix]
    var_prediction_all_asset_lasso_fix[, asset_index] = var_prediction_asset[,index_lasso_fix]
    
    asset_index = asset_index + 1
  }
  
  tick_loss_per_lambda_lasso_per_RW[index_RW,] = colMeans(tick_loss_per_lambda_lasso)
  tick_loss_per_lambda_ridge_per_RW[index_RW,] = colMeans(tick_loss_per_lambda_ridge)
  tick_loss_per_lambda_elastic_per_RW[index_RW,] = colMeans(tick_loss_per_lambda_elastic)
  
  
  write.csv(var_prediction_all_asset_lasso_bic, paste(output_path,"Lasso_bic_",RW,"RW.csv"))
  write.csv(var_prediction_all_asset_ridge_cv, paste(output_path,"Ridge_cv_",RW,"RW.csv"))
  write.csv(var_prediction_all_asset_elastic_cv, paste(output_path,"ELastic_cv_",RW,"RW.csv"))
  write.csv(var_prediction_all_asset_lasso_cv, paste(output_path,"Lasso_cv_",RW,"RW.csv"))
  write.csv(var_prediction_all_asset_ridge_fix, paste(output_path,"Ridge_fix_",RW,"RW.csv"))
  write.csv(var_prediction_all_asset_elastic_fix, paste(output_path,"Elastic_fix_",RW,"RW.csv"))
  write.csv(var_prediction_all_asset_lasso_fix, paste(output_path,"Lasso_fix_",RW,"RW.csv"))
  
  index_RW = index_RW + 1
}

write.csv(tick_loss_per_lambda_lasso_per_RW, paste(output_path,"avg_tick_loss_lasso"))
write.csv(tick_loss_per_lambda_ridge_per_RW, paste(output_path,"avg_tick_loss_ridge"))
write.csv(tick_loss_per_lambda_elastic_per_RW, paste(output_path,"avg_tick_loss_elastic"))


tick_loss<-function(rdt, var, var_quantile){
  hit = var_quantile*sweep(rdt,-var,1, FUN =  "<") * sweep(rdt, var,1, FUN = "-")
  result = (t(hit) %*% sweep(rdt,var,1,FUN = "-"))/length(rdt)
  
  return (result)
}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# on utilise bic a minimiser en fct de lambda
# valable pour le lasso
BIC <-function(rdt, var, alpha, betas){
  t <- length(rdt)
  df <- sum(betas!=0)
  result <- log(tick_loss(rdt, var, alpha)) + (log(t)/2*t)*df
  return (result)
}


#%%%%%%%%%%%%%%%%%%%%
# Cross Validation
cross_validation<-function(rdt, var, var_quantile){
  hit = var_quantile*sweep(rdt,-var,1, FUN =  "<") * sweep(rdt, var,1, FUN = "-")
  result = (t(hit) %*% sweep(rdt,var,1,FUN = "-"))/(length(rdt)-1000)
  
  return (result)
}


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#heuristic rule
heuristic<-function(lambda, betas){
  s = 0.8
  max_lambda = 0
  for (i in length(lambda)){
    if (rowSums(betas)[i]<0.8 & max_lambda < lambda[i] ){
      max_lambda = lambda[i]
      max_index = i
    }
  }
  #result<- list(max_lambda, max_index)
  return(max_index)
}




