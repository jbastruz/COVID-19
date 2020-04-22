#List of packages required
knitr::opts_chunk$set(echo = TRUE)

Packages=c("rstanarm","actuar","rstan","dplyr","lubridate","readxl","ggplot2",
           "dplyr","brms","bayesplot","tidyverse","shinystan","gganimate",
           "gifski","png","lazyeval","formattable","scales","DT","tidybayes",
           "ggrepel","ggstance","ggridges","purrr","forcats","shiny","shinythemes",
           "MASS","EnvStats","hrbrthemes","reshape2","psych","lattice","grid","ggthemes", "plotly", "broom"
)

lapply(Packages,library,character.only=TRUE)

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
Sys.setenv(LOCAL_CPPFLAGS = '-march=corei7')

plotListInit <- function(A, B) {
  lapply(seq_len(A), function(x) {
    
    fig <- plot_ly(B[B$Country.Region == Country[x],], x = ~day, color = ~I("rgba(7, 164, 181, 1)"), name = Country[x]) 
    fig <- fig %>% add_markers(y = ~cumcases, text = rownames(B[B$Country.Region == Country[x],]), showlegend = FALSE)
    fig <- fig %>% add_text(x = 40, y = 1.1*max(1,1*B[B$Country.Region == Country[x],]$cumcases), text = ~unique(Country[x]), color = I("black"))
    fig <- fig %>% layout(showlegend = FALSE, title = "DataViz For Gross Data (Confirmed Cases)", xaxis = list(showgrid = FALSE), yaxis = list(showgrid = FALSE))
  }
  )}

plotList <- function(A, B) {
  lapply(seq_len(A), function(x) {
    
    fig <- plot_ly(B[B$Country.Region == Country[x],], x = ~day, color = ~I("black"), name = Country[x]) 
    fig <- fig %>% add_markers(y = ~cumcases, text = rownames(B[B$Country.Region == Country[x],]), showlegend = FALSE)
    fig <- fig %>% add_text(x = 40, y = max(B[B$Country.Region == Country[x],]$Y_pred_cred_0.975), text = ~unique(Country[x]), color = I("black"))
    fig <- fig %>% add_lines(y = ~Y_pred_median,
                             line = list(color = 'rgba(7, 164, 181, 1)'))
    fig <- fig %>% add_ribbons(data = B[B$Country.Region == Country[x],],
                               ymin = ~Y_pred_cred_0.025,
                               ymax = ~Y_pred_cred_0.975,
                               line = list(color = 'rgba(7, 164, 181, 0.05)'),
                               fillcolor = 'rgba(7, 164, 181, 0.2)')
    
    fig <- fig %>% layout(showlegend = FALSE, xaxis = list(showgrid = FALSE), yaxis = list(showgrid = FALSE))
  }
)}


theme.1 <- theme(axis.line = element_line(size = 3, colour = "grey80"), 
                 axis.text = element_text(size=12),
                 legend.box.background = element_rect(),
                 legend.text=element_text(size=12),
                 axis.title=element_text(size=13,face="bold"),
                 panel.background = element_rect(fill = "white", colour = "grey50"),
                 legend.box.margin = margin(6, 6, 6, 6))

theme_Publication <- function(base_size=14, base_family="helvetica") {
  (theme_foundation(base_size=base_size, base_family=base_family)
    + theme(plot.title = element_text(face = "bold",
                                      size = rel(1.2), hjust = 0.5),
            text = element_text(),
            panel.background = element_rect(colour = NA),
            plot.background = element_rect(colour = NA),
            panel.border = element_rect(colour = NA),
            axis.title = element_text(face = "bold",size = rel(1)),
            axis.title.y = element_text(angle=90,vjust =2),
            axis.title.x = element_text(vjust = -0.2),
            axis.text = element_text(), 
            axis.line = element_line(colour="black"),
            axis.ticks = element_line(),
            panel.grid.major = element_line(colour="#f0f0f0"),
            panel.grid.minor = element_blank(),
            legend.key = element_rect(colour = NA),
            legend.position = "bottom",
            legend.direction = "horizontal",
            legend.key.size= unit(0.2, "cm"),
            legend.margin = unit(0, "cm"),
            legend.title = element_text(face="italic"),
            plot.margin=unit(c(10,5,5,5),"mm"),
            strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
            strip.text = element_text(face="bold")
    ))
  
}

devPlot <- function(x, data,
                    main="Confirmed cases development by regions",
                    ylab = "Confirmed cases",
                    xlab="Days",...){

  key <- list(rep=FALSE, columns=2,
              lines=list(col=c("#00526D", "#00526D"), 
                         type=c("p", "p"), 
                         pch=c(19, 1)),
              text=list(lab=c("Observation", "Hold out observation")))
  xyplot(x, data, t=c("b", "b"), pch=c(19, 1), 
         col=c("#00526D", "#00526D"),
         layout=c(5,2),
         par.settings = list(strip.background=list(col="#CBDDE6")),
         par.strip.text = list(font = 2),
         key = key, 
         as.table=TRUE,
         scales=list(alternating=1),
         xlab=xlab, ylab=ylab, main=main,
         sub="Kaggle_count_cases")
}

plotDevBananas <- function(x, data,
                           xlab="Confirmed cases development by regions", 
                           ylab="Confirmed cases", 
                           main="Gompertz model",
                           ...){
  key <- list(
    rep=FALSE, 
    lines=list(col=c("#00526D", "#00526D", "purple"), 
               type=c("p", "p", "l"), 
               pch=c(19, 1, NA)),
    text=list(lab=c("Observation", "Hold out observation", "Mean estimate")),
    rectangles = list(col=adjustcolor("yellow", alpha.f=0.5), border="grey"),
    text=list(lab="95% Posterior prediction interval"))
  
  xyplot(x, data=data,as.table=TRUE,xlab=xlab, ylab=ylab, main=main,
         sub="Kaggle_data",
         scales=list(alternating=1), layout=c(5,2), key=key,
         par.settings = list(strip.background=list(col="#CBDDE6")),
         par.strip.text = list(font = 2),
         panel=function(x, y){
           n <- length(x)
           divisor <- 5
           cn <- c(1:(n/divisor))
           upper <- y[cn+n/divisor*0]
           lower <- y[cn+n/divisor*1]
           x <- x[cn]
           panel.polygon(c(x, rev(x)), c(upper, rev(lower)),
                         col = adjustcolor("yellow", alpha.f = 0.5), 
                         border = "grey")
           panel.lines(x, y[cn+n/divisor*2], col="purple")
           panel.points(x, y[cn+n/divisor*4], lwd=1, col="#00526D")
           panel.points(x, y[cn+n/divisor*3], lwd=1, pch=19, col="#00526D")
         }, ...)
  
  
}

createB <- function(stanfit, data, probs=c(0.025, 0.975)){
  
  post <- as.matrix(rstan::extract(stanfit, pars=c("yrep","yproj"),inc_warmup=FALSE))
  ppc_summary <- cbind(
    mean=apply(post, 2, mean),
    t(apply(post, 2, quantile, probs=probs))
  )
  colnames(ppc_summary) = c(
    "Y_pred_mean",
    paste0("Y_pred_cred", gsub("\\.", "", probs[1])),
    paste0("Y_pred_cred", gsub("\\.", "", probs[2])))
  
  return(cbind(ppc_summary, data))
}


split_two_factors <- function(Data,Factor1,Factor2,Cost,Exposure){
  tmp<- Data %>% 
    group_by_(Factor1,Factor2) %>%
    summarise_(Cost= interp(~sum(var1), var1=as.name(Cost)),
               #Npatients= interp(~sum(var1), var1=as.name(Npatients)),
               Exposure= interp(~sum(var2), var2=as.name(Exposure))) %>%
    #mutate(Frequency=ClaimsNumber/Exposure) %>%
    #mutate(AverageCost= Costs/ClaimsNumber) %>%
    mutate(PurePremium= Cost/Exposure)
  tmp<- data.frame(tmp)
  return(tmp)
}



split_three_factors <- function(Data,Factor1,Factor2,Factor3,Cost,Exposure){
  tmp<- Data %>% 
    group_by_(Factor1,Factor2,Factor3) %>%
    summarise_(Cost= interp(~sum(var1), var1=as.name(Cost)),
               #Npatients= interp(~sum(var1), var1=as.name(Npatients)),
               Exposure= interp(~sum(var2), var2=as.name(Exposure))) %>%
    #mutate(Frequency=ClaimsNumber/Exposure) %>%
    #mutate(AverageCost= Costs/ClaimsNumber) %>%
    mutate(PurePremium= Cost/Exposure)
  tmp<- data.frame(tmp)
  return(tmp)
}

split_four_factors <- function(Data,Factor1,Factor2,Factor3,Factor4,Cost,Exposure){
  tmp<- Data %>% 
    group_by_(Factor1,Factor2,Factor3,Factor4) %>%
    summarise_(Cost= interp(~sum(var1), var1=as.name(Cost)),
               #Npatients= interp(~sum(var1), var1=as.name(Npatients)),
               Exposure= interp(~sum(var2), var2=as.name(Exposure))) %>%
    #mutate(Frequency=ClaimsNumber/Exposure) %>%
    #mutate(AverageCost= Costs/ClaimsNumber) %>%
    mutate(PurePremium= Cost/Exposure)
  tmp<- data.frame(tmp)
  return(tmp)
}

graph_factor_line<- function(Data,x_axis,y_axis,color){
  
  ggplot(Data,aes_string(x_axis,y_axis,group=color))+
    geom_point(aes_string(color=color),size=3)+
    theme_gray()+
    theme(axis.text.x = element_text(angle = 45, vjust = 0.6,size=13),
          axis.text.y = element_text(size=12),
          axis.title = element_text(size=13),
          legend.text = element_text(size=12),
          legend.background = element_rect(fill="gray90", size=0.5))+
    geom_line(aes_string(color=color),size=0.8)
}

graph_factor_point<- function(Data,x_axis,y_axis,color,size){
  
  ggplot(Data,aes_string(x_axis,y_axis))+
    geom_point(aes_string(color=color,size=size))+
    scale_size_continuous(labels=comma)+
    theme_gray()+
    theme(axis.text.x = element_text(vjust = 0.6,size=13),
          axis.text.y = element_text(size=12),
          axis.title = element_text(size=13),
          legend.text = element_text(size=12),
          legend.background = element_rect(fill="gray90", size=0.5))+
    guides(colour = guide_legend(override.aes = list(size=3)))
    #scale_x_continuous(breaks = seq(0,1200,200))
}

graph_x_continuous_animated<- function(Data,x_axis,y_axis,size,color){
  
  ggtmp<-ggplot(Data, aes_string(y=y_axis,x=x_axis)) +
    geom_point(aes_string(color=color, size=size))+
    theme_gray()+
    theme(axis.text.x = element_text(size=13),
          axis.text.y = element_text(size=13),
          legend.text = element_text(size=13))+
    scale_size_continuous(labels=comma)+
    scale_x_continuous(breaks = seq(0,1200,100))+
    transition_time(yr)+
    labs(title = 'Year: {frame_time}', x = x_axis, y = y_axis) +
    ease_aes('cubic-in-out')+ 
    guides(colour = guide_legend(override.aes = list(size=5)))
  
  animate(ggtmp,nframes=140,duration=15,width = 1000, height = 500, end_pause = 40)
  
}

graph_y_continuous_animated<- function(Data,x_axis,y_axis,size,color){
  
  ggtmp<-ggplot(Data, aes_string(y=y_axis,x=x_axis)) +
    geom_point(aes_string(color=color, size=size))+
    theme_gray()+
    theme(axis.text.x = element_text(size=13),
          axis.text.y = element_text(size=13),
          legend.text = element_text(size=13))+
    scale_size_continuous(labels=comma)+
    scale_y_continuous(breaks=seq(0,1000,100))+
    transition_time(yr)+
    labs(title = 'Year: {frame_time}', x = x_axis, y = y_axis) +
    ease_aes('cubic-in-out')+ 
    guides(colour = guide_legend(override.aes = list(size=5)))
  
  animate(ggtmp,nframes=140,duration=15,width = 1000, height = 500, end_pause = 40)
  
}

graph_x_continuous_split_sc_animated<- function(Data,x_axis,y_axis,size,color){
  
  Data$yr<-as.integer(Data$yr)
  ggtmp<-ggplot(Data, aes_string(y=y_axis,x=x_axis)) +
    geom_point(aes_string(color=color, size=size))+
    theme_gray()+
    theme(axis.text.x = element_text(size=13),
          axis.text.y = element_text(size=11),
          legend.text = element_text(size=13))+
    scale_size_continuous(labels=comma)+
    scale_x_continuous(breaks = seq(0,1200,100))+
    transition_time(yr)+
    labs(title = 'Year: {frame_time}', x = x_axis, y = y_axis) +
    ease_aes('cubic-in-out')+ 
    guides(colour = guide_legend(override.aes = list(size=5)))+
    facet_wrap(~service_category)
  
  animate(ggtmp,nframes=140,duration=15,width = 1200, height = 500, end_pause = 40)
  
}

posteriorcheck<-function(Data,Model,Factor){
  fit_yrep<-posterior_predict(Model)
  dim(fit_yrep)
  windows();pp_check(Model,nsamples = 60)
  windows();ppc_stat_grouped(Data$AllowedPMPM,fit_yrep,group = f1,
                             stat="mean")
  windows();ppc_stat_grouped(Data$AllowedPMPM,fit_yrep,group = f1,
                             stat="sd")
}


sim_claimcost_gamma<-function(Data,mu,disp){
  tmp_cc<-rep(NA,nrow(Data))
  for (i in 1:nrow(Data)){
    tmp_cn<-Data[i,"NC_Outpatient"]
    tmp_cc[i]<-sum(rgammaAlt(tmp_cn,mu,disp))
  }
  Data$CC_Outpatient<-tmp_cc
  return(Data)
}

sim_claimcost_invgaussian<-function(Data,mu,disp){
  tmp_cc<-rep(NA,nrow(Data))
  for (i in 1:nrow(Data)){
    tmp_cn<-Data[i,"NC_Outpatient"]
    tmp_cc[i]<-sum(rinv_gaussian(tmp_cn,mu,disp))
  }
  Data$CC_Outpatient<-tmp_cc
  return(Data)
}

sim_claimcost_lognormal<-function(Data,mu,disp){
  tmp_cc<-rep(NA,nrow(Data))
  for (i in 1:nrow(Data)){
    tmp_cn<-Data[i,"NC_Outpatient"]
    tmp_cc[i]<-sum(rlnorm(tmp_cn,mu,disp))
  }
  Data$CC_Outpatient<-tmp_cc
  return(Data)
}

