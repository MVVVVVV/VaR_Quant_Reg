library(rqPen)
library(hqreg)
library(data.table)
library(tibble)


tick_loss<-function(rdt, var, var_quantile){
  
  hit = (var_quantile-sweep(-var,1,-rdt, FUN =  "<"))*sweep(-var,1,rdt,FUN="+")
  result = colSums(hit /length(rdt))
  
  return (result)
}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# on utilise bic a minimiser en fct de lambda
# valable pour le lasso
BIC <-function(rdt, var, alpha, betas){
  
  betas[is.na(betas)] = 0
  t <- length(rdt)
  df <- colSums(betas!=0)
  result <- log(tick_loss(rdt, var, alpha)) + (log(t)/(2*t))*df
  return (result)
}


#%%%%%%%%%%%%%%%%%%%%
# Cross Validation
cross_validation<-function(rdt, var, var_quantile){
  
  hit = (var_quantile-sweep(-var,1,-rdt, FUN =  "<"))*sweep(-var,1,rdt,FUN="+")
  result = colSums(hit/(length(rdt)-1000))
  
  return (result)
}


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#heuristic rule
heuristic<-function(lambda, betas){
  
  
  betas[is.na(betas)] = 0
  s = 0.8
  max_lambda = 0
  for (i in length(lambda)){
    if (sum(betas[,i])<0.8 & max_lambda < lambda[i] ){
      max_lambda = lambda[i]
      max_index = i
    }
  }
  if (max_lambda==0){
    max_index = 1
  }
  #result<- list(max_lambda, max_index)
  return(max_index)
}

test_debug<-function(){
  var_quantile = 0.01
  max_sum_betas = 1
  rolling_windows = c("ERW")
  start_date = "2011-01-03"
  end_date = "2012-12-31"
  my_path = "C:\\Users\\MV\\Desktop\\Projet Quant\\"
  input_assets_path = "C:\\Users\\MV\\Documents\\GitHub\\VaR_Quant_Reg\\Assets\\"
  output_path = "C:\\Users\\MV\\Documents\\GitHub\\VaR_Quant_Reg\\Results RQ\\"
  
  rdt <- read.csv(paste(my_path,"rdt.csv", sep=""), header=TRUE, row.names="Date")
  #%%%%%%%%%%%%%
  
  start_index = grep(start_date, row.names(rdt))
  end_index = grep(end_date, row.names(rdt))
  nbr_prediction = end_index - start_index +1
  
  n = 1
  
  # on crée nos lambdas sur echelle de 10
  log_values <- cbind(0.1,1,10,100,1000,10000,100000)
  nb_steps = 15
  lambdas = seq(0,1,1/(nb_steps))*0.01
  
  
  for (i in 1:length(log_values)){
   temp = seq(0,1,1/(nb_steps))*log_values[i]
   lambdas = append(lambdas, temp[2:length(temp)])
  }
  # on itere sur nos 5 rolling windows
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
    var_prediction_all_asset_lasso_cv = matrix(, nrow=nbr_prediction, ncol = 30)
    var_prediction_all_asset_ridge_fix = matrix(, nrow=nbr_prediction, ncol = 30)
    var_prediction_all_asset_elastic_fix = matrix(, nrow=nbr_prediction, ncol = 30)
    var_prediction_all_asset_lasso_fix = matrix(, nrow=nbr_prediction, ncol = 30)
    asset_index = 13
    
    for(asset in colnames(rdt)[14:length(colnames(rdt))]){
      
      
      var = read.csv(paste(input_assets_path,"models_asset_",(asset_index-1),".csv", sep=""))
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
      date_index = 1
      
      for (date_row in seq(start_index,end_index, by=1)){
        
        
        if (RW =="ERW"){
          start_rol_win = 1
        }else{
          start_rol_win = date_row-as.numeric(RW)
        }
        
        end_rol_win = date_row - 1

        my_rdt = rdt[start_rol_win:end_rol_win, asset]
        
        date_var = row.names(rdt)[end_rol_win]
        date_var_end= grep(date_var, var[,1])
        
        
        if (RW =="ERW"){
          date_var_beg = 1
        }else{
          date_var_beg = date_var_end-as.numeric(RW) +1
        }
        
       
        my_rdt = as.numeric(unlist(my_rdt))
        my_var = data.matrix(var[date_var_beg:date_var_end,2:length(var)])
        
        fit_lasso = hqreg( my_var, my_rdt, method = "quantile", tau = var_quantile, alpha = 1, 
                          lambda = lambdas, screen = "ASR", max.iter = 100)
        fit_ridge = hqreg(my_var, my_rdt, method = "quantile", tau = var_quantile, alpha = 0,
                          lambda = lambdas, screen = "ASR", max.iter = 100)
        fit_elastic = hqreg(my_var, my_rdt, method = "quantile", tau = var_quantile, alpha = 0.5, 
                            lambda = lambdas, screen = "ASR", max.iter = 100)
        
        fit_elastic$beta[is.na(fit_elastic$beta)] = 0
        fit_lasso$beta[is.na(fit_lasso$beta)] = 0
        fit_ridge$beta[is.na(fit_ridge$beta)] = 0
        
        
        arg_x = as.numeric( var[date_var_end+1,2:ncol(var)])
        
        var_prediction_lasso[date_index,] = predict(fit_lasso, arg_x,
                                                  lambda = lambdas)
        var_prediction_ridge[date_index,] = predict(fit_ridge, arg_x,
                                                  lambda = lambdas)
        var_prediction_elastic[date_index,] = predict(fit_elastic, arg_x,
                                                    lambda = lambdas)
        
        if (date_index%%100 == 0){
          print(date_index)
          t1 <- Sys.time()
          print(t1-t0)
        }

        date_index = date_index + 1
        
      }
      
      
      
      tick_loss_per_lambda_lasso[asset_index,] =  tick_loss(rdt[start_index:end_index, asset_index], var_prediction_lasso, var_quantile)
      tick_loss_per_lambda_ridge[asset_index,] = tick_loss(rdt[start_index:end_index, asset_index], var_prediction_ridge, var_quantile)
      tick_loss_per_lambda_elastic[asset_index,] = tick_loss(rdt[start_index:end_index, asset_index], var_prediction_elastic, var_quantile)
      
      
      
      
      print(asset)
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
                                 betas=fit_ridge$beta[2:18,] )
      index_elastic_fix =heuristic(lambda=fit_elastic$lambda,
                                   betas=fit_elastic$beta[2:18,] )
      index_lasso_fix =heuristic(lambda=fit_lasso$lambda,
                                 betas=fit_lasso$beta[2:18,] )
      
      
      
      # on ajoute l'asset pour chaque modele
      var_prediction_all_asset_lasso_bic[, asset_index] = var_prediction_lasso[,index_lasso_bic]
      var_prediction_all_asset_ridge_cv[, asset_index] = var_prediction_ridge[,index_ridge_cv]
      var_prediction_all_asset_elastic_cv[, asset_index] = var_prediction_elastic[,index_elastic_cv]
      var_prediction_all_asset_lasso_cv[, asset_index] = var_prediction_lasso[,index_lasso_cv]
      var_prediction_all_asset_ridge_fix[, asset_index] = var_prediction_ridge[,index_ridge_fix]
      var_prediction_all_asset_elastic_fix[, asset_index] = var_prediction_elastic[,index_elastic_fix]
      var_prediction_all_asset_lasso_fix[, asset_index] = var_prediction_lasso[,index_lasso_fix]
      
      asset_index = asset_index + 1
      
      write.csv(var_prediction_all_asset_lasso_bic, paste(output_path,"Lasso_bic_",
                                                          toString(RW),"RW.csv", sep=""))
      write.csv(var_prediction_all_asset_ridge_cv, paste(output_path,"Ridge_cv_",
                                                         toString(RW),"RW.csv", sep=""))
      write.csv(var_prediction_all_asset_elastic_cv, paste(output_path,"ELastic_cv_",
                                                           toString(RW),"RW.csv", sep=""))
      write.csv(var_prediction_all_asset_lasso_cv, paste(output_path,"Lasso_cv_",
                                                         toString(RW),"RW.csv", sep=""))
      write.csv(var_prediction_all_asset_ridge_fix, paste(output_path,"Ridge_fix_",
                                                          toString(RW),"RW.csv", sep=""))
      write.csv(var_prediction_all_asset_elastic_fix, paste(output_path,"Elastic_fix_",
                                                            toString(RW),"RW.csv", sep=""))
      write.csv(var_prediction_all_asset_lasso_fix, paste(output_path,"Lasso_fix_",
                                                          toString(RW),"RW.csv", sep=""))
      
      write.csv(tick_loss_per_lambda_lasso, paste(output_path,"avg_tick_loss_lasso.csv", sep=""))
      write.csv(tick_loss_per_lambda_ridge, paste(output_path,"avg_tick_loss_ridge.csv", sep=""))
      write.csv(tick_loss_per_lambda_elastic, paste(output_path,"avg_tick_loss_elastic.csv", sep=""))
      
      
    }
    
    
    tick_loss_per_lambda_lasso_per_RW[index_RW,] = colMeans(tick_loss_per_lambda_lasso)
    tick_loss_per_lambda_ridge_per_RW[index_RW,] = colMeans(tick_loss_per_lambda_ridge)
    tick_loss_per_lambda_elastic_per_RW[index_RW,] = colMeans(tick_loss_per_lambda_elastic)
    
    
    write.csv(var_prediction_all_asset_lasso_bic, paste(output_path,"Lasso_bic_",
                                                        toString(RW),"RW.csv", sep=""))
    write.csv(var_prediction_all_asset_ridge_cv, paste(output_path,"Ridge_cv_",
                                                       toString(RW),"RW.csv", sep=""))
    write.csv(var_prediction_all_asset_elastic_cv, paste(output_path,"ELastic_cv_",
                                                         toString(RW),"RW.csv", sep=""))
    write.csv(var_prediction_all_asset_lasso_cv, paste(output_path,"Lasso_cv_",
                                                       toString(RW),"RW.csv", sep=""))
    write.csv(var_prediction_all_asset_ridge_fix, paste(output_path,"Ridge_fix_",
                                                        toString(RW),"RW.csv", sep=""))
    write.csv(var_prediction_all_asset_elastic_fix, paste(output_path,"Elastic_fix_",
                                                          toString(RW),"RW.csv", sep=""))
    write.csv(var_prediction_all_asset_lasso_fix, paste(output_path,"Lasso_fix_",
                                                        toString(RW),"RW.csv", sep=""))
    
    index_RW = index_RW + 1
  }
  
  
  write.csv(tick_loss_per_lambda_lasso_per_RW, paste(output_path,"avg_tick_loss_lasso.csv", sep=""))
  write.csv(tick_loss_per_lambda_ridge_per_RW, paste(output_path,"avg_tick_loss_ridge.csv", sep=""))
  write.csv(tick_loss_per_lambda_elastic_per_RW, paste(output_path,"avg_tick_loss_elastic.csv", sep=""))
}

