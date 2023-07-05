# This code was originally created by Giora Simchoni and Saharon Rosset and modified by Fabio Sigrist, 2023

library(janitor)

# Note: data_cleaned_train_comments_X.csv is the result of an ETL process described in Kalehbasti et. al. (2019).
# We followed the script in their Github repo exactly.
data_all <- read_csv("../raw_data/airbnb_data.zip", name_repair = "minimal")
coeffs <- as.vector(read_csv("../raw_data/selected_coefs.txt", col_names=FALSE)[,1])$X1
for(i in 1:length(coeffs)) {
  coeffs[i] <- str_replace(coeffs[i], "\x92", "’")
}
data <- data_all[,c("host_id","price",coeffs)]
colnames(data) <- janitor::make_clean_names(colnames(data))
data %>% write_csv("../data/airbnb.csv.gz")

