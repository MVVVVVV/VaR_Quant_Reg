
####installation des librairies nécessaires:

installed.packages("hqreg")
library(hqreg)
install.packages("readxl")
install.packages('writexl')
library(readxl)
library(writexl)
install.packages("data.table")
library(data.table)
install.packages("SparseM")
library(SparseM)

######################################################################################
#Unpenalized QR:
#######################################################################"



#lecture du fichier de données à partir des chemins:

input_assets_path = ""
my_path = ""

#le fichier rdt vous est envoyé par mail

rdt <- read.csv(paste(my_path,"rdt.csv", sep=""), header=TRUE, row.names="Date")


################


#création de la matrice des predictions

forec<-(matrix(nrow=3772,ncol=2))
pred<-(matrix(nrow=3773,ncol=30))

#boucle sur tous les actifs:

for(j in 2:31){

   #lecture du fichier des VaR pour chaque actif pour tous les modèles de prédiction (standalone forecasts)

   var <- read.csv(paste(input_assets_path,"models_asset_",(j-2),".csv", sep=""),header=TRUE, row.names="Date")
   VAR<-data.matrix(var, rownames.force = NA)
   
   #sélection du vecteur des rendements de l'actif et transormation en vecteur
   
    my_rdt <- rdt[1:3771, j-1]
    my_rdt <- as.vector(data.matrix(my_rdt, rownames.force = NA))
    
    #Estimation  du modèle grace au package hqreg en utilisant la méthode quantile correspondant à notre fonction de perte 
    #ici nous fixons lambda comme étant le vecteur (0,1) mais ne récupérons que les VaR correspondant à lambda=0
    
    fit_qr<-hqreg(VAR,my_rdt,method="quantile",tau=0.01,lambda=cbind(0,1))
    
    #prévisions des VaR en utilisant les betas estimés
    
    forec<-predict(fit_qr,VAR,lambda = cbind(0,1),type="response")
    pred[,j-1]=forec[,1]
    
    # reset de la matrice forec
    
    forec<-matrix(nrow=3771,ncol=2)
    
  
}

#création du fichier des VaR:

df<-data.frame(pred)
write.csv(df,"QR unpenalized.csv")





############################################################################################
#2% Convex Quantile regression; 
#######################################################################"

#mettres les chemins 

input_assets_path = ""
my_path = ""
rdt <- read.csv(paste(my_path,"rdt.csv", sep=""), header=TRUE, row.names="Date")


#création des contraines, terme de gauche et terme de droite

n <- 2014
p <- 16

#matrice des contraintes

R <- rbind(diag(p+1), -diag(p+1))
r <- c(rep( 0, p+1), rep(-1, p+1))
sR1<-as.matrix.csr(R)

#date de début et date de fin

start_date = "2007-01-03"
end_date = "2014-12-31"
start_index = grep(start_date, row.names(rdt))
end_index = grep(end_date, row.names(rdt))

#création de la matrice pour stocker les VaR estimées aavec les betas

prediction <- matrix(,ncol = 30, nrow = 2014)

#boucle sur les 30 actifs

for(j in 2:31){
  
  #lectue du fichier de VaR pour chaque actif
  
  var <- read.csv(paste(input_assets_path,"models_asset_",(j-2),".csv", sep=""),header=TRUE,row.names = "dates")
  var<-as.matrix(var)
  
  #Ajouter un vecteur de 1 pour le cefficient Beta0: intercept
  
 VAR_<-cbind(1,var)

    #boucle sur les dates: fenetre glissante
  
    for (t in seq(start_index,end_index)){ 
      start_rw = 1
      end_rw = t 
      
      #récupération du vecteur des rendements
      my_rdt <- rdt[1:(t-1),j-1]
      my_rdt <- as.vector(data.matrix(my_rdt, rownames.force = NA))
      
      
      date_var = row.names(rdt)[end_rw]
      date_var_end = grep(date_var,row.names(VAR_))-2
      my_var = (VAR_[1:date_var_end, 1:18])
      
      #transformation de la matrice my_var en objet csr requis pour la fonction rq.fit.sfnc
      
      VAR_CSR<-as.matrix.csr(my_var)
      
      my_rdt<-as.numeric(my_rdt)
      
      #estimation DES coeffcients avec les contraintes
      
      fit_Convex_qr<-rq.fit.sfnc(VAR_CSR,my_rdt,sR1,r,tau=0.01)
      co<-fit_Convex_qr
      prediction[t,j-1] = co[1] + co[2:18]%*%my_var[t,2:18]
      
      
    }
  
 
  
}


  