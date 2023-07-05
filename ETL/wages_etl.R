# This code was originally created by Fabio Sigrist, 2023

# Set working directory to this files directory
library(tidyverse)
library(haven)#  for read_dta
library(fastDummies)

## source: https://www.stata-press.com/data/r9/xtmain.html
data <- as.data.frame(read_dta("../raw_data/nlswork.dta")) 

covars <- c("grade", "age", "ttl_exp", "tenure", "not_smsa",
            "south", "year", "msp", "nev_mar", "collgrad",
            "c_city", "hours","ind_code", "occ_code", "race")
low_card_cat_vars <- c("ind_code", "occ_code", "race", "year")
for (var in low_card_cat_vars) {
  data[,var] <- as.factor(data[,var])
  if (sum(is.na(data[,var])) > 0) {
    data[,var] <- addNA(data[,var])
  }
}

data <- data[,c(covars,"ln_wage","idcode")]
# For simplicity, exclude NAs
nas <- apply(is.na(data),1,any)
# sum(nas)
data <- data[!nas,]

# Add dummies for low-cardinality categorical variables
data <- fastDummies::dummy_cols(data, remove_first_dummy = TRUE)
data$t = as.numeric(as.character(data$year)) - 78
data <- data[,-which(names(data) %in% low_card_cat_vars)]

data %>% write_csv("../data/wages.csv.gz")

