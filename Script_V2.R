source("Utilities.R")

setwd("C:/Users/juan.deoyarbide/Documents/GitHub/COVID-19")

confirmedcases_counts <- read.csv("https://data.humdata.org/hxlproxy/data/download/time_series_covid19_confirmed_global_narrow.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&merge-replace02=on&merge-overwrite02=on&filter03=explode&explode-header-att03=date&explode-value-att03=value&filter04=rename&rename-oldtag04=%23affected%2Bdate&rename-newtag04=%23date&rename-header04=Date&filter05=rename&rename-oldtag05=%23affected%2Bvalue&rename-newtag05=%23affected%2Binfected%2Bvalue%2Bnum&rename-header05=Value&filter06=clean&clean-date-tags06=%23date&filter07=sort&sort-tags07=%23date&sort-reverse07=on&filter08=sort&sort-tags08=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv")[-1,]

confirmedcases_counts$cumcases <- as.numeric(levels(confirmedcases_counts$Value)[confirmedcases_counts$Value])

death_counts <- read.csv("https://data.humdata.org/hxlproxy/data/download/time_series_covid19_deaths_global_narrow.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&merge-replace02=on&merge-overwrite02=on&filter03=explode&explode-header-att03=date&explode-value-att03=value&filter04=rename&rename-oldtag04=%23affected%2Bdate&rename-newtag04=%23date&rename-header04=Date&filter05=rename&rename-oldtag05=%23affected%2Bvalue&rename-newtag05=%23affected%2Binfected%2Bvalue%2Bnum&rename-header05=Value&filter06=clean&clean-date-tags06=%23date&filter07=sort&sort-tags07=%23date&sort-reverse07=on&filter08=sort&sort-tags08=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv")[-1,]

countryinfo <- read_csv("countryinfo.csv")

confirmedcases_counts$Country.Region <- as.character(confirmedcases_counts$Country.Region)
death_counts$Country.Region <- as.character(death_counts$Country.Region)

confirmedcases_counts[confirmedcases_counts$Country.Region=="Korea, South",c("Country.Region")] <- "South Korea"
death_counts[death_counts$Country.Region=="Korea, South",c("Country.Region")] <- "South Korea"

confirmedcases_counts$day <- as.numeric(as.Date(confirmedcases_counts$Date))-min(as.numeric(as.Date(confirmedcases_counts$Date)))+1
death_counts$day <- as.numeric(as.Date(death_counts$Date))-min(as.numeric(as.Date(death_counts$Date)))+1

tmp <- left_join(confirmedcases_counts %>% 
                   group_by(Country.Region,date(Date)) %>% 
                   filter(Value!=0) %>% 
                   group_by(Country.Region) %>% 
                   summarize(firstDate=min(date(Date)),
                             firstDay=min(day)),
                 countryinfo)

tmp$lockdown <- as.Date(max(date(confirmedcases_counts$Date)))

tmp[tmp$Country.Region=="Italy",c("lockdown")] <- as.Date("2020/03/09")
tmp[tmp$Country.Region=="Spain",c("lockdown")] <- as.Date("2020/03/14")
tmp[tmp$Country.Region=="France",c("lockdown")] <- as.Date("2020/03/17")
tmp[tmp$Country.Region=="Germany",c("lockdown")] <- as.Date("2020/03/22")
tmp[tmp$Country.Region=="United Kingdom",c("lockdown")] <- as.Date("2020/03/23")
tmp[tmp$Country.Region=="Austria",c("lockdown")] <- as.Date("2020/03/16")
tmp[tmp$Country.Region=="Belgium",c("lockdown")] <- as.Date("2020/03/18")
tmp[tmp$Country.Region=="Portugal",c("lockdown")] <- as.Date("2020/03/19")
tmp[tmp$Country.Region=="China",c("lockdown")] <-as.Date("2020/01/23")
tmp[tmp$Country.Region=="US",c("lockdown")] <- as.Date("2020/03/22")
tmp[tmp$Country.Region=="India",c("lockdown")] <- as.Date("2020/03/24")
tmp[tmp$Country.Region=="Argentina",c("lockdown")] <- as.Date("2020/03/20")
# tmp[tmp$Country.Region=="Luxembourg",c("lockdown")] <- as.Date("2020/03/19")
# tmp[tmp$Country.Region=="Canada",c("lockdown")] <- as.Date("2020/03/19")
# tmp[tmp$Country.Region=="Mexico",c("lockdown")] <- as.Date("2020/03/31")
# tmp[tmp$Country.Region=="Chile",c("lockdown")] <- as.Date("2020/03/23")
# tmp[tmp$Country.Region=="Uruguay",c("lockdown")] <- as.Date("2020/03/22")
# tmp[tmp$Country.Region=="Brazil",c("lockdown")] <- as.Date("2020/03/23")
# tmp[tmp$Country.Region=="Russia",c("lockdown")] <- as.Date("2020/03/10")
# tmp[tmp$Country.Region=="Taiwan*",c("lockdown")] <- as.Date("2020/03/3")
# tmp[tmp$Country.Region=="Vietnam",c("lockdown")] <- as.Date("2020/03/5")

tmp$lockdown_delay <- as.numeric(tmp$lockdown)-as.numeric(tmp$firstDate)

db_level_1 <- confirmedcases_counts %>% 
  group_by(Country.Region,day) %>% 
  summarize(cumcases= sum(cumcases)) %>%  
  filter(Country.Region %in% c("France","Germany","China")
         )

#"India","South Korea","Taiwan*","Canada","Mexico","Chile","Uruguay","Brazil","Argentina","Luxembourg","Austria","Russia","Portugal",
# c("China",
# "US",
# "France",
# "Italy",
# "Spain",
# "Germany",
# "Belgium",
# "United Kingdom")
# )


db_level_1$Country.Region <- as.factor(db_level_1$Country.Region)
db_level_1$Country.Region <- relevel(db_level_1$Country.Region,"China")


tmp_2 <- countryinfo %>% 
  filter(Country.Region %in% c("France","Germany","China")
         )

ordered_names <- data.frame(expand.grid(levels(db_level_1$Country.Region)))
colnames(ordered_names) <- "Country.Region"

db_level_2 <- left_join(ordered_names,tmp_2)

db_level_2 <- left_join(db_level_2,tmp[,c(1,21)])

ndaysproj=10

yproj <- expand.grid(Country.Region=levels(db_level_1$Country.Region),
                     day=max(db_level_1$day+1):(max(db_level_1$day+1)+ndaysproj))

design_matrix_1 <- model.matrix(~-1+Country.Region,db_level_1);colnames(design_matrix_1);dim(design_matrix_1)

design_matrix_b <- model.matrix(~-1+Country.Region,db_level_1)[,-1];colnames(design_matrix_b);dim(design_matrix_b)

design_matrix_level_2 <- model.matrix(~ -1 + as.numeric(lockdown_delay)+
                                        as.numeric(TestsOver1mPop)+
                                        as.numeric(Density)+
                                        Group+
                                        as.numeric(Urbanpop),
                                      db_level_2);colnames(design_matrix_level_2);dim(design_matrix_level_2)

design_matrix_proj <- model.matrix(~-1+Country.Region,yproj);colnames(design_matrix_proj);dim(design_matrix_proj)

design_matrix_b_proj <- model.matrix(~-1+Country.Region,yproj)[,-1];colnames(design_matrix_b_proj);dim(design_matrix_b_proj)

db_level_1$cumcases <- db_level_1$cumcases+1

stan_list <- list("y"=db_level_1$cumcases,
                  "N"=length(db_level_1$cumcases),
                  "K"=dim(design_matrix_1)[2],
                  "K_c"=dim(design_matrix_level_2)[2],
                  "design_matrix_1"=design_matrix_1,
                  "design_matrix_b"=design_matrix_b,
                  "design_matrix_level_2"=design_matrix_level_2,
                  "design_matrix_proj"=design_matrix_proj,
                  "design_matrix_b_proj"=design_matrix_b_proj,
                  "M"=length(yproj$Country.Region),
                  "day"=db_level_1$day,
                  "day_projected"=yproj$day)

stan_list <- list("y"=db_level_1$cumcases,
                  "N"=length(db_level_1$cumcases),
                  "K"=dim(design_matrix_1)[2],
                  "design_matrix_1"=design_matrix_1,
                  "design_matrix_b"=design_matrix_b,
                  "design_matrix_proj"=design_matrix_proj,
                  "design_matrix_b_proj"=design_matrix_b_proj,
                  "M"=length(yproj$Country.Region),
                  "day"=db_level_1$day,
                  "day_projected"=yproj$day)

stan_fit<- stan(
  file="stan_code_V2.stan",
  data=stan_list,
  control = list(adapt_delta=0.9, max_treedepth=10),
  chains=3,
  iter =6000,
  warmup = 2000
)

traceplot(stan_fit,pars=c("b_intercept","c_a","c_b","c_h",
                          "a","b","sigma_b",
                          "sigma"))

pairs(stan_fit,pars=c("a_intercept","c_intercept","beta_c"))


post <- rstan::extract(stan_fit,pars = c("yrep","yproj"),
                       inc_warmup=FALSE)

ppc_summary_yrep <- data.frame(cbind(
  median=apply(post$yrep, 2, median),
  t(apply(post$yrep, 2, quantile, probs=c(0.025, 0.975)))
))

colnames(ppc_summary_yrep) = 
  c("Y_pred_median","Y_pred_cred_0.025","Y_pred_cred_0.975")

ppc_summary_yrep$model <- "yrep"

ppc_summary_yproj <- data.frame(cbind(
  median=apply(post$yproj, 2, median),
  t(apply(post$yproj, 2, quantile, probs=c(0.025, 0.975)))
))

colnames(ppc_summary_yproj) = 
  c("Y_pred_median","Y_pred_cred_0.025","Y_pred_cred_0.975")

ppc_summary_yproj$model <- "yproj"

ppc_summary <- rbind(ppc_summary_yrep,ppc_summary_yproj)

c2 = data.frame(rstan::extract(stan_fit,par = "c_2"))
a = data.frame(rstan::extract(stan_fit,par = "a"))
g1 = mcmc_combo(c2)
g2 = mcmc_intervals(a)

ppc_intervals_grouped(y=c(db_level_1$cumcases,ppc_summary_yproj$Y_pred_median),
                      yrep=cbind(post$yrep,post$yproj),
                      x=c(db_level_1$day,yproj$day),
                      group=c(as.character(db_level_1$Country.Region),as.character(yproj$Country.Region)))+
  ggplot2::geom_vline(xintercept=85,linetype="dashed")

A=300000
b=10
c=0.001+1:200*0.001 #-(1:200)^2*0.001
t=1:200

sim=A*exp(-b * exp(-c * t))

ggplot(data.frame(sim=sim,t),aes(y=sim,x=t))+
  geom_line()






