## Code for reproducing the results of Sigrist (2023) "A Comparison of Machine Learning Methods for Data with High-Cardinality Categorical Variables"
## Author: Fabio Sigrist, 03.07.2023

# setwd(dirname(rstudioapi::getSourceEditorContext()$path))
train_dir_catboost = paste0(getwd(),"/")

library(tidyverse)
library(lme4)
library(rsample)
library(recipes)
library(glue)
library(gpboost)
# library(devtools) # for installation of catboost
# devtools::install_url('https://github.com/catboost/catboost/releases/download/v1.1.1/catboost-R-Windows-1.1.1.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
library(catboost)

# Data sets
single_categorical_var_data_sets <- c("airbnb")
multiple_categorical_vars_data_sets <- c("imdb", "spotify", "news", "inst_eval")
longitudinal_data_sets <- c("rossmann", "au_anual_import_commodity", "wages")
all_data_sets <- c(single_categorical_var_data_sets, 
                   multiple_categorical_vars_data_sets, 
                   longitudinal_data_sets)

use_saved_tuning_parameters <- FALSE
# Set "use_saved_tuning_parameters <- TRUE", if tuning parameters have already 
#   been chosen and the experiments should be run again

for(name_dataset in all_data_sets) {
  
  # Load data
  if (name_dataset != "inst_eval") {
    df <- read_csv(paste0("data/",name_dataset,".csv.gz"))
  }
  
  # Specify fixed effects predictor variables, response variable, and categorical variables
  if (name_dataset == "imdb") {
    colnames(df) <- janitor::make_clean_names(colnames(df))
    x_cols <- colnames(df)[-which(colnames(df) %in% c("director_id", "type_id", "score"))]
    label <- "score"
    cat_vars <- c("director_id", "type_id")
    # form_lme4 <- as.formula(str_c("score ~ ", str_c(str_c(x_cols, collapse = " + "), 
    #                                                 " + (1 | director_id) + (1 | type_id)")))
  } else if (name_dataset == "spotify") {
    x_cols <- colnames(df)[-which(colnames(df) %in% c("artist_id", "album_id", "playlist_id", 
                                                      "subgenre_id", "danceability",
                                                      'track_id', 'track_artist', 'track_album_id', 
                                                      'track_album_release_date', 'pl_subgenres', 'playlist_ids'))]
    label <- "danceability"
    cat_vars <- c("artist_id", "album_id", "playlist_id", "subgenre_id")
    # form_lme4 <- as.formula(str_c("danceability ~ ", str_c(str_c(x_cols, collapse = " + "),
    #                                                        " + (1 | artist_id) + (1 | album_id) + (1 | playlist_id) + (1 | subgenre_id)")))
  } else if (name_dataset == "news") {
    df$Facebook <- log(df$Facebook + 1)
    x_cols <- colnames(df)[-which(colnames(df) %in% c("title_id", "source_id", "Facebook",
                                                      "GooglePlus", "LinkedIn", "IDLink", "Title",
                                                      "Headline", "Source", "Topic", "PublishDate"))]  
    label <- "Facebook"
    cat_vars <- c("title_id", "source_id")
    # form_lme4 <- as.formula(str_c("Facebook ~ ", str_c(str_c(x_cols, collapse = " + "), 
    #                                                    " + (1 | title_id) + (1 | source_id)")))
  } else if (name_dataset == "inst_eval") {
    df <- InstEval
    for (j in 1:dim(df)[2]) {
      df[,j] <- as.numeric(df[,j])
    }
    df <- as_tibble(df)
    x_cols <- colnames(df)[-which(colnames(df) %in% c("s", "d", "dept", "y"))]
    label <- "y"
    cat_vars <- c("s", "d", "dept")
    # form_lme4 <- as.formula(str_c("y ~ ", str_c(str_c(x_cols, collapse = " + "), 
    #                                             " + (1 | s) + (1 | d) + (1 | dept)")))
  } else if (name_dataset == "rossmann") {
    x_cols <- colnames(df)[-which(colnames(df) %in% c("date", "Sales", "Store", "year"))]
    label <- "Sales"
    cat_vars <- "Store"
    # form_lme4 <- as.formula(str_c("Sales ~ ", str_c(str_c(x_cols, collapse = " + "), 
    #                                                 " + + (1 |Store) + (0 + t|Store) + (0 + I(t^2)|Store)")))
  } else if (name_dataset == "au_anual_import_commodity") {
    df$trade_usd <- log(df$trade_usd)
    x_cols <- colnames(df)[-which(colnames(df) %in% c("commodity", "commodity_id", "trade_usd", "comm_code", "year"))]
    label <- "trade_usd"
    cat_vars <- "commodity_id"
    # form_lme4 <- as.formula(str_c("trade_usd ~ ", str_c(str_c(x_cols, collapse = " + "),
    #                                                     " + (1 |commodity_id) + (0 + t|commodity_id) + (0 + I(t^2)|commodity_id)")))
  } else if (name_dataset == "airbnb") {
    x_cols <- colnames(df)[-which(colnames(df) %in% c("host_id", "price"))]
    label <- "price"
    cat_vars <- "host_id"
    # form_lme4 <- as.formula(str_c("price ~ ", str_c(str_c(x_cols, collapse = " + "), 
    #                                                 " + (1 | host_id)")))
  } else if (name_dataset == "wages") {
    x_cols <- colnames(df)[-which(colnames(df) %in% c("ln_wage", "idcode", "t"))]
    label <- "ln_wage"
    cat_vars <- "idcode"
    # form_lme4 <- as.formula(str_c("ln_wage ~ ", str_c(str_c(x_cols, collapse = " + "),
    #                                                   " + I(age^2) + I(ttl_exp^2) + I(tenure^2) + 
    #                                                   (1 |idcode) + (0 + t|idcode) + (0 +I(t^2)|idcode)")))
  }
  
  # Create cross-validation (CV) splits
  ncv <- 5
  set.seed(100)
  split_obj <- vfold_cv(df, v = ncv)
  # Save CV splits
  train_ids <- list()
  for (k in 1:ncv) {
    train_ids[[k]] <- split_obj$splits[[k]]$in_id
  }
  write(RJSONIO::toJSON(train_ids, pretty = TRUE), paste0("cv_folds/",name_dataset,".json"))
  # Check equality
  # train_ids_load <- RJSONIO::fromJSON(content=paste0("cv_folds/",name_dataset,".json"), simplifyWithNames = FALSE)
  # k <- 2
  # sum(abs(train_ids_load[[k]] - train_ids[[k]]))
  
  # For saving results
  res_list_gpb <- res_list_gpb_excl <- res_list_gpb_incl <- res_list_boost_cat  <-
    res_list_boost_cont <- res_list_catboost <- res_list_linear <- list()
  
  # Run cross-validation
  for (k in 1:ncv) {
    
    cat(glue(" CV iteration: {k}"), "\n")
    df_train <- analysis(split_obj$splits[[k]])
    df_test <- assessment(split_obj$splits[[k]])
    
    # Prepare data and models for fitting and evaluation
    y_train <- df_train[,label][[1]]
    y_test <- df_test[,label][[1]]
    X_lin_train <- cbind(Intcpt=rep(1, length(y_train)), as.matrix(df_train[,x_cols]))
    X_lin_test <- cbind(Intcpt=rep(1, length(y_test)), as.matrix(df_test[,x_cols]))
    dataset <- gpb.Dataset(data = as.matrix(df_train[,x_cols]), label = y_train)
    dataset_boost_cont <- gpb.Dataset(data = as.matrix(df_train[,c(cat_vars,x_cols)]), 
                                      label = y_train)
    dataset_boost_cat <- gpb.Dataset(data = as.matrix(df_train[,c(cat_vars,x_cols)]), 
                                     label = y_train,
                                     categorical_feature = 1:length(cat_vars))
    data_pred <- as.matrix(df_test[,x_cols])
    data_pred_boost <- as.matrix(df_test[,c(cat_vars,x_cols)])
    group_data = df_train[,cat_vars]
    group_data_pred = df_test[,cat_vars]
    group_rand_coef_data <- NULL
    group_rand_coef_data_pred <- NULL
    ind_effect_group_rand_coef <- NULL
    if (name_dataset == "rossmann") {
      group_rand_coef_data = cbind(df_train$t, df_train$t^2)
      group_rand_coef_data_pred = cbind(df_test$t, df_test$t^2)
      ind_effect_group_rand_coef = c(1,1)
    } else if (name_dataset == "au_anual_import_commodity") {
      group_rand_coef_data = cbind(df_train$t, df_train$t^2)
      group_rand_coef_data_pred = cbind(df_test$t, df_test$t^2)
      ind_effect_group_rand_coef = c(1,1)
    } else if (name_dataset == "wages") {
      group_rand_coef_data = cbind(df_train$t, df_train$t^2)
      group_rand_coef_data_pred = cbind(df_test$t, df_test$t^2)
      ind_effect_group_rand_coef = c(1,1)
    }
    # Preparation for choosing tuning parameters
    param_grid = list("learning_rate" = c(1,0.1,0.01), 
                      "min_data_in_leaf" = c(10,100,1000),
                      "max_depth" = c(1,2,3,5,10),
                      "lambda_l2" = c(0,1,10))
    other_params <- list(objective = "regression_l2", metric = "mse", num_leaves = 2^10)
    n <- dim(df_train)[1]
    set.seed(k)
    valid_tune_idx <- sample.int(n, as.integer(0.2*n))
    folds = list(valid_tune_idx)
    
    ###################
    ## Linear mixed effects model
    ###################
    cat("\n")
    print(paste0("********* Starting linear mixed effects model (",name_dataset,", k = ", k,")"))
    params <- list()
    if (!(name_dataset %in% single_categorical_var_data_sets)) {
      params = list(optimizer_cov="nelder_mead", maxit = 5000) ## It is often faster to use nelder_mead since gradient calculation is slow for multi-level grouped RE models
    }
    start <- Sys.time()
    lin_gp_model <- fitGPModel(group_data = group_data, likelihood = "gaussian",
                               group_rand_coef_data = group_rand_coef_data,
                               ind_effect_group_rand_coef = ind_effect_group_rand_coef,
                               y = y_train, X = X_lin_train, params = params)
    end <- Sys.time()
    y_pred <- lin_gp_model$predict(group_data_pred = group_data_pred,
                                   group_rand_coef_data_pred = group_rand_coef_data_pred,
                                   X_pred = X_lin_test, predict_response=TRUE)$mu

    if (name_dataset == "spotify") {## apply same clipping as Simchoni and Rosset (2023)
      y_pred <- ifelse(y_pred < 0, 0, ifelse(y_pred > 1, 1, y_pred))
    }
    # Same thing with lme4
    # start <- Sys.time()
    # lme_mod <- lmer(form_lme4, df_train)
    # end <- Sys.time()
    # y_pred <- predict(lme_mod, df_test, allow.new.levels = TRUE, type = "response")
    error <- mean((y_test - y_pred)^2)
    res_list_linear[[k]] <- list( experiment = k - 1, exp_type = "linear",
                                  error = error, time = difftime(end, start, units="secs"))

    ###################
    ## GPBoost not including categorical variables in fixed effects
    ###################
    cat("\n")
    print(paste0("********* Starting gpboost_excl (",name_dataset,", k = ", k,")"))
    gp_model <- GPModel(group_data = group_data, likelihood = "gaussian",
                        group_rand_coef_data = group_rand_coef_data,
                        ind_effect_group_rand_coef = ind_effect_group_rand_coef)
    if (!(name_dataset %in% single_categorical_var_data_sets)) {
      gp_model$set_optim_params(params = list(optimizer_cov="nelder_mead")) ## It is often faster to use nelder_mead since gradient calculation is slow for multi-level grouped RE models
    }
    # Choosing tuning parameters
    if (use_saved_tuning_parameters) {
      opt_params <- RJSONIO::fromJSON(content=paste0("tune_pars/",name_dataset,"_gpboost__fold=",k,".json"),
                                      simplifyWithNames = FALSE)
    } else {
      set.seed(k*2)
      start <- Sys.time()
      opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = other_params,
                                                    num_try_random = NULL, folds = folds,
                                                    data = dataset, gp_model = gp_model,
                                                    use_gp_model_for_validation=TRUE, verbose_eval = 1,
                                                    nrounds = 1000, early_stopping_rounds = 10)
      end <- Sys.time()
      opt_params$time = difftime(end, start, units="secs")
      write(RJSONIO::toJSON(opt_params, pretty = TRUE),
            paste0("tune_pars/",name_dataset,"_gpboost__fold=",k,".json"))
    }
    # Train model
    params <- list(num_leaves = 2^10,
                   learning_rate = opt_params$best_params$learning_rate,
                   max_depth = opt_params$best_params$max_depth,
                   lambda_l2 = opt_params$best_params$lambda_l2)
    nrounds = opt_params$best_iter
    start <- Sys.time()
    gpb <- gpb.train(data = dataset, gp_model = gp_model, nrounds = nrounds,
                     params = params, verbose = 0)
    end <- Sys.time()
    # Make predictions
    y_pred <- predict(gpb, data = data_pred, group_data_pred = group_data_pred,
                      group_rand_coef_data_pred = group_rand_coef_data_pred,
                      predict_var = FALSE, pred_latent = FALSE)$response_mean
    if (name_dataset == "spotify") {## apply same clipping as Simchoni and Rosset (2023)
      y_pred <- ifelse(y_pred < 0, 0, ifelse(y_pred > 1, 1, y_pred))
    }
    error <- mean((y_test - y_pred)^2)
    res_list_gpb_excl[[k]] <- list( experiment = k - 1, exp_type = "gpboost_excl",
                                    error = error, time = difftime(end, start, units="secs"))

    ###################
    ## GPBoost including categorical variables in fixed effects
    ###################
    cat("\n")
    print(paste0("********* Starting gpboost_incl (",name_dataset,", k = ", k,")"))
    gp_model <- GPModel(group_data = group_data, likelihood = "gaussian",
                        group_rand_coef_data = group_rand_coef_data,
                        ind_effect_group_rand_coef = ind_effect_group_rand_coef)
    if (!(name_dataset %in% single_categorical_var_data_sets)) {
      gp_model$set_optim_params(params = list(optimizer_cov="nelder_mead")) ## It is often faster to use nelder_mead since gradient calculation is slow for multi-level grouped RE models
    }
    # Choosing tuning parameters
    if (use_saved_tuning_parameters) {
      opt_params <- RJSONIO::fromJSON(content=paste0("tune_pars/",name_dataset,"_gpboost_incl__fold=",k,".json"),
                                      simplifyWithNames = FALSE)
    } else {
      set.seed(k*2)
      start <- Sys.time()
      opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = other_params,
                                                    num_try_random = NULL, folds = folds,
                                                    data = dataset_boost_cat, gp_model = gp_model,
                                                    use_gp_model_for_validation=TRUE, verbose_eval = 1,
                                                    nrounds = 1000, early_stopping_rounds = 10)
      end <- Sys.time()
      opt_params$time = difftime(end, start, units="secs")
      write(RJSONIO::toJSON(opt_params, pretty = TRUE),
            paste0("tune_pars/",name_dataset,"_gpboost_incl__fold=",k,".json"))
    }
    # Train model
    params <- list(num_leaves = 2^10,
                   learning_rate = opt_params$best_params$learning_rate,
                   max_depth = opt_params$best_params$max_depth,
                   lambda_l2 = opt_params$best_params$lambda_l2)
    start <- Sys.time()
    gpb <- gpb.train(data = dataset_boost_cat, gp_model = gp_model, nrounds = opt_params$best_iter,
                     params = params, verbose = 0)
    end <- Sys.time()
    # Make predictions
    y_pred <- predict(gpb, data = data_pred_boost, group_data_pred = group_data_pred,
                      group_rand_coef_data_pred = group_rand_coef_data_pred,
                      predict_var = FALSE, pred_latent = FALSE)$response_mean
    if (name_dataset == "spotify") {## apply same clipping as Simchoni and Rosset (2023)
      y_pred <- ifelse(y_pred < 0, 0, ifelse(y_pred > 1, 1, y_pred))
    }
    error <- mean((y_test - y_pred)^2)
    res_list_gpb_incl[[k]] <- list( experiment = k - 1, exp_type = "gpboost_incl",
                                    error = error, time = difftime(end, start, units="secs"))
    
    ###################
    ## GPBoost: considering "including or not including categorical variables in fixed effects" as tuning option
    ###################
    cat("\n")
    print(paste0("********* Starting gpboost (",name_dataset,", k = ", k,")"))
    gp_model <- GPModel(group_data = group_data, likelihood = "gaussian",
                        group_rand_coef_data = group_rand_coef_data,
                        ind_effect_group_rand_coef = ind_effect_group_rand_coef)
    if (!(name_dataset %in% single_categorical_var_data_sets)) {
      gp_model$set_optim_params(params = list(optimizer_cov="nelder_mead")) ## It is often faster to use nelder_mead since gradient calculation is slow for multi-level grouped RE models
    }
    # Decide whether to include or exclude categorical variables in fixed effects
    opt_params_excl <- RJSONIO::fromJSON(content=paste0("tune_pars/",name_dataset,"_gpboost__fold=",k,".json"),
                                         simplifyWithNames = FALSE)
    opt_params_incl <- RJSONIO::fromJSON(content=paste0("tune_pars/",name_dataset,"_gpboost_incl__fold=",k,".json"),
                                         simplifyWithNames = FALSE)
    if (opt_params_excl$best_score < opt_params_incl$best_score) {
      opt_params <- opt_params_excl
      data_gpb <- dataset
      data_gpb_pred <- data_pred
    } else {
      opt_params <- opt_params_incl
      data_gpb <- dataset_boost_cat
      data_gpb_pred <- data_pred_boost
    }
    # Train model
    params <- list(num_leaves = 2^10,
                   learning_rate = opt_params$best_params$learning_rate,
                   max_depth = opt_params$best_params$max_depth,
                   lambda_l2 = opt_params$best_params$lambda_l2)
    start <- Sys.time()
    gpb <- gpb.train(data = data_gpb, gp_model = gp_model, nrounds = opt_params$best_iter,
                     params = params, verbose = 0)
    end <- Sys.time()
    # Make predictions
    y_pred <- predict(gpb, data = data_gpb_pred, group_data_pred = group_data_pred,
                      group_rand_coef_data_pred = group_rand_coef_data_pred,
                      predict_var = FALSE, pred_latent = FALSE)$response_mean
    if (name_dataset == "spotify") {## apply same clipping as Simchoni and Rosset (2023)
      y_pred <- ifelse(y_pred < 0, 0, ifelse(y_pred > 1, 1, y_pred))
    }
    error <- mean((y_test - y_pred)^2)
    res_list_gpb[[k]] <- list( experiment = k - 1, exp_type = "gpboost",
                               error = error, time = difftime(end, start, units="secs"))
    
    ###################
    ## Tree-boosting treating categorical variables as continuous ones
    ###################
    cat("\n")
    print(paste0("********* Starting boost_cont (",name_dataset,", k = ", k,")"))
    # Choosing tuning parameters
    if (use_saved_tuning_parameters) {
      opt_params <- RJSONIO::fromJSON(content=paste0("tune_pars/",name_dataset,"_boost_cont_fold=",k,".json"),
                                      simplifyWithNames = FALSE)
    } else {
      set.seed(k*2)
      start <- Sys.time()
      opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = other_params,
                                                    num_try_random = NULL, folds = folds,
                                                    data = dataset_boost_cont, verbose_eval = 1,
                                                    nrounds = 1000, early_stopping_rounds = 10)
      end <- Sys.time()
      opt_params$time = difftime(end, start, units="secs")
      write(RJSONIO::toJSON(opt_params, pretty = TRUE),
            paste0("tune_pars/",name_dataset,"_boost_cont_fold=",k,".json"))
    }
    # Train model
    params <- list(objective = "regression_l2", num_leaves = 2^10,
                   learning_rate = opt_params$best_params$learning_rate,
                   max_depth = opt_params$best_params$max_depth,
                   lambda_l2 = opt_params$best_params$lambda_l2)
    start <- Sys.time()
    bst <- gpb.train(data = dataset_boost_cont, nrounds = opt_params$best_iter,
                     params = params, verbose = 0)
    end <- Sys.time()
    # Make predictions
    y_pred <- predict(bst, data = data_pred_boost, pred_latent = FALSE)
    if (name_dataset == "spotify") {## apply same clipping as Simchoni and Rosset (2023)
      y_pred <- ifelse(y_pred < 0, 0, ifelse(y_pred > 1, 1, y_pred))
    }
    error <- mean((y_test - y_pred)^2)
    res_list_boost_cont[[k]] <- list( experiment = k - 1, exp_type = "boost_cont",
                                      error = error, time = difftime(end, start, units="secs"))

    ###################
    ## Tree-boosting with categorical variable approach of LightGBM
    ###################
    cat("\n")
    print(paste0("********* Starting boost_cat (",name_dataset,", k = ", k,")"))
    # Choosing tuning parameters
    if (use_saved_tuning_parameters) {
      opt_params <- RJSONIO::fromJSON(content=paste0("tune_pars/",name_dataset,"_boost_cat_fold=",k,".json"),
                                      simplifyWithNames = FALSE)
    } else {
      set.seed(k*2)
      start <- Sys.time()
      opt_params <- gpb.grid.search.tune.parameters(param_grid = param_grid, params = other_params,
                                                    num_try_random = NULL, folds = folds,
                                                    data = dataset_boost_cat, verbose_eval = 1,
                                                    nrounds = 1000, early_stopping_rounds = 10)
      end <- Sys.time()
      opt_params$time = difftime(end, start, units="secs")
      write(RJSONIO::toJSON(opt_params, pretty = TRUE),
            paste0("tune_pars/",name_dataset,"_boost_cat_fold=",k,".json"))
    }
    # Train model
    params <- list(objective = "regression_l2", num_leaves = 2^10,
                   learning_rate = opt_params$best_params$learning_rate,
                   max_depth = opt_params$best_params$max_depth,
                   lambda_l2 = opt_params$best_params$lambda_l2)
    start <- Sys.time()
    bst <- gpb.train(data = dataset_boost_cat, nrounds = opt_params$best_iter,
                     params = params, verbose = 0)
    end <- Sys.time()
    # Make predictions
    y_pred <- predict(bst, data = data_pred_boost, pred_latent = FALSE)
    if (name_dataset == "spotify") {## apply same clipping as Simchoni and Rosset (2023)
      y_pred <- ifelse(y_pred < 0, 0, ifelse(y_pred > 1, 1, y_pred))
    }
    error <- mean((y_test - y_pred)^2)
    res_list_boost_cat[[k]] <- list( experiment = k - 1, exp_type = "boost_cat",
                                     error = error, time = difftime(end, start, units="secs"))

    ###################
    ## Tree-boosting with categorical variable approach of CatBoost
    ###################
    cat("\n")
    print(paste0("********* Starting catboost (",name_dataset,", k = ", k,")"))
    # Choosing tuning parameters
    if (use_saved_tuning_parameters) {
      opt_params <- RJSONIO::fromJSON(content=paste0("tune_pars/",name_dataset,"_catboost_fold=",k,".json"),
                                      simplifyWithNames = FALSE)
    } else {
      params <- opt_params <- list()
      best_nrounds <- 1000
      counter_num_comb <- 1
      num_param_comps <- gpboost:::get.grid.size(param_grid)
      best_score <- 1E99
      start <- Sys.time()
      for (param_comb_number in 1:num_param_comps) {
        param_comb = gpboost:::get.param.combination(param_comb_number=param_comb_number,
                                                     param_grid=param_grid)
        param_comb_str <- lapply(seq_along(param_comb), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=param_comb, n=names(param_comb))
        param_comb_str <- paste0(unlist(param_comb_str), collapse=", ")
        message(paste0("Trying parameter combination ", counter_num_comb,
                       " of ", num_param_comps, ": ", param_comb_str, " ..."))
        for (param in names(param_comb)) {
          params[[param]] <- param_comb[[param]]
        }
        dtrain_catboost = catboost.load_pool(as.matrix(df_train[folds[[1]], c(cat_vars,x_cols)]),
                                             label = y_train[folds[[1]]], cat_features =  c(0:(length(cat_vars)-1)))
        dval_catboost = catboost.load_pool(as.matrix(df_train[-folds[[1]], c(cat_vars,x_cols)]),
                                           label = y_train[-folds[[1]]], cat_features =  c(0:(length(cat_vars)-1)))
        fit_params <- list(iterations = 1000, border_count = 255, l2_leaf_reg = params$lambda_l2,
                           depth = params$max_depth, learning_rate = params$learning_rate,
                           min_data_in_leaf = params$min_data_in_leaf, loss_function = "RMSE", eval_metric = "RMSE",
                           early_stopping_rounds = 10, logging_level="Silent", train_dir=train_dir_catboost)
        model <- catboost.train(dtrain_catboost, dval_catboost, params = fit_params)
        test_score <- as.numeric(read.table(paste0(train_dir_catboost,"test_error.tsv"),header=TRUE)[,2])
        if (min(test_score) < best_score) {
          best_score <- min(test_score)
          best_iter <- which.min(test_score)
          opt_params <- param_comb
          param_comb_print <- param_comb
          param_comb_print[["nrounds"]] <- best_iter
          str <- lapply(seq_along(param_comb_print), function(y, n, i) { paste0(n[[i]],": ", y[[i]]) }, y=param_comb_print, n=names(param_comb_print))
          message(paste0("***** New best test score (",
                         best_score,  ") found for the following parameter combination: ",
                         paste0(unlist(str), collapse=", ")))
        }
        counter_num_comb <- counter_num_comb + 1L
      }
      end <- Sys.time()
      opt_params$time = difftime(end, start, units="secs")
      opt_params$best_iter <- best_iter
      opt_params$best_score <- best_score
      write(RJSONIO::toJSON(opt_params, pretty = TRUE),
            paste0("tune_pars/",name_dataset,"_catboost_fold=",k,".json"))
    }
    # Train model
    dtrain_catboost = catboost.load_pool(as.matrix(df_train[, c(cat_vars,x_cols)]),
                                         label = y_train, cat_features = c(0:(length(cat_vars)-1)))
    fit_params <- list(iterations = opt_params$best_iter, border_count = 255, l2_leaf_reg = opt_params$lambda_l2,
                       depth = opt_params$max_depth, learning_rate = opt_params$learning_rate,
                       min_data_in_leaf = opt_params$min_data_in_leaf, logging_level="Silent",
                       loss_function = "RMSE", train_dir=train_dir_catboost)
    start <- Sys.time()
    model <- catboost.train(dtrain_catboost, params = fit_params)
    end <- Sys.time()
    # Make predictions
    y_pred <- catboost.predict(model, catboost.load_pool(as.matrix(df_test[,c(cat_vars,x_cols)]),
                                                         cat_features = c(0:(length(cat_vars)-1))),
                               prediction_type = "RawFormulaVal")
    if (name_dataset == "spotify") {## apply same clipping as Simchoni and Rosset (2023)
      y_pred <- ifelse(y_pred < 0, 0, ifelse(y_pred > 1, 1, y_pred))
    }
    error <- mean((y_test - y_pred)^2)
    res_list_catboost[[k]] <- list( experiment = k - 1, exp_type = "catboost",
                                    error = error, time = difftime(end, start, units="secs"))
    
    
    # Save results
    (res_df_linear <- bind_rows(res_list_linear))
    (res_df_gpboost <- bind_rows(res_list_gpb))
    (res_df_gpboost_excl <- bind_rows(res_list_gpb_excl))
    (res_df_gpboost_incl <- bind_rows(res_list_gpb_incl))
    (res_df_boost_cont <- bind_rows(res_list_boost_cont))
    (res_df_boost_cat <- bind_rows(res_list_boost_cat))
    (res_df_catboost <- bind_rows(res_list_catboost))
    
    write_csv(res_df_linear, paste0("results/res_",name_dataset,"_linear.csv"))
    write_csv(res_df_gpboost, paste0("results/res_",name_dataset,"_gpboost.csv"))
    write_csv(res_df_gpboost_excl, paste0("results/res_",name_dataset,"_gpboost_excl.csv"))
    write_csv(res_df_gpboost_incl, paste0("results/res_",name_dataset,"_gpboost_incl.csv"))
    write_csv(res_df_boost_cont, paste0("results/res_",name_dataset,"_boost_cont.csv"))
    write_csv(res_df_boost_cat, paste0("results/res_",name_dataset,"_boost_cat.csv"))
    write_csv(res_df_catboost, paste0("results/res_",name_dataset,"_catboost.csv"))
    
  }# end loop over CV folds 
  
}# end loop over all_data_sets
