#installation des packages n�cessaires

install.packages("readxl")
install.packages('writexl')
install.packages("rugarch")
library(readxl)
library(writexl)
library(rugarch)

#lecture des donn�es � partir du fichier de donn�es fourni

my_path = ""

#le fichier rdt vous est envoy� par mail
rdt <- read.csv(paste(my_path,"rdt.csv", sep=""), header=TRUE, row.names="Date")

rdt<-data.frame(data)

# compter le nombre de lignes et de colonnes des donn�es
ligne<-nrow(data)
col<-ncol(data)

#####################################################################
#specification du mod�le APARCH Normal

for (j in 2:col){
  
  garchSpec <- ugarchspec(variance.model=list(model="apARCH",garchOrder=c(1,1)),mean.model=list(armaOrder=c(0,0)), distribution.model="norm")
  garchFit <- ugarchfit(spec=garchSpec,rdt[j] )
  coef(garchFit)
  mod = ugarchroll(garchSpec, rdt[j], n.ahead = 1, n.start = 1000,refit.every = 1000, refit.window = "moving",window.size = 1000, solver = "hybrid", fit.control = list(),calculate.VaR = TRUE, VaR.alpha = 0.01,keep.coef = TRUE)
  var<-as.data.frame(mod,which=4)
}


  
###############################################################
#specification du mod�le APARCH Student


for (j in 2:ncol){
  garchSpec <- ugarchspec(variance.model=list(model="apARCH",garchOrder=c(1,1)),mean.model=list(armaOrder=c(1,1)), distribution.model="std")
  garchFit <- ugarchfit(spec=garchSpec,rdt[j] )
  coef(garchFit)
  mod = ugarchroll(garchSpec, rdt[j], n.ahead = 1, n.start = 1000,refit.every = 1, refit.window = "moving",window.size = 1000, solver = "hybrid", fit.control = list(),calculate.VaR = TRUE, VaR.alpha = 0.01,keep.coef = TRUE)
  var<-as.data.frame(mod,which=4)
}
  

#########################################################################################
#specification du mod�le APARCH FHS:

#d�finition de la matrice de stockage:
VaR_FHS<-matrix(nrow=ligne)


for (j in 2:ncol){
  
  
  garchSpec_FHS<- ugarchspec(variance.model=list(model="apARCH",garchOrder=c(1,1)),mean.model=list(armaOrder=c(1,1)), distribution.model="norm")
  garchFit_FHS <- ugarchfit(spec=garchSpec_FHS, rdt[j])
  pred<-ugarchroll(garchSpec_FHS, rdt[j], n.ahead = 1, n.start = 1000,refit.every = 1000, refit.window = "moving",window.size = 1000, solver = "hybrid", fit.control = list(),calculate.VaR = FALSE,keep.coef = TRUE)
  garch_sigma <- pred@forecast[["density"]][["Sigma"]]
  
  #calcul des r�sidus standardis�s
  
  res_std<- residuals(garchFit_FHS)/sigma(garchFit_FHS)
  q<-quantile(res_std,0.01)
  VaR_FHS<-garch_sigma*q
  
}

