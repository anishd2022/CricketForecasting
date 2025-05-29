

options(width=540, stringsAsFactors=FALSE)

library(mactivate)

projpath <- getwd()

xpath_data <- file.path(projpath, "AD_Cric/_data")

#print(
#load(file.path(xpath_data, "cleanedData.RData"))
#)

print(
load(file.path(xpath_data, "cleanedDataSort.RData"))
)

xobatted -> xbatted
xomx_runs_filled -> xmx_runs_filled
xomx_wickets_filled -> xmx_wickets_filled
xodf_key_use -> xdf_key_use



################

N <- length(xbatted)

xndx_vt <- seq(1, N, by=2)
xndx_ht <- seq(2, N, by=2)

xbatted_bin <- 2 * xbatted - 1

#### target

xtarget_runs <- rep(0, N)

xtarget_runs[ xndx_ht ] <- xmx_runs_filled[ xndx_vt, ncol(xmx_runs_filled) ]

Nall <- N



#########################

#set.seed(777)
#xndx_test <- sample(1:Nall, size=4000)
#xndx_train <- setdiff( 1:Nall, xndx_test )


Ntest <- 4000

### xndx_test <- sample((Nall-Ntrain+1):Nall, size=4000)
xndx_test <- (Nall-Ntest+1):Nall

xndx_train <- setdiff( 1:Nall, xndx_test )


#######################################
#######################################
#######################################
#######################################




#######################################

xmore_than_2W_remain <- 2 * (xmx_wickets_filled[ , c("W90") ] <= 8) - 1

Xall <-
cbind(
#"int"=rep(1, length(xbatted_bin)),
xbatted_bin,
xtarget_runs,
xmore_than_2W_remain,
xmx_runs_filled[ , c("R36", "R48", "R66", "R78", "R90") ],
xmx_wickets_filled[ , c("W36", "W48", "W66", "W78", "W90") ]
)

yall <-  xmx_runs_filled[ , c("R120") ]




xmean_train <- apply(Xall[ xndx_train, ], 2, mean)
xsd_train   <- apply(Xall[ xndx_train, ], 2, sd)



Xstnd <- t( ( t(Xall) - xmean_train ) / xsd_train )

yy <- yall


########### here
U_train <- Xstnd[ xndx_train, !(colnames(Xstnd) %in% "int"), drop=FALSE ]
U_test  <- Xstnd[ xndx_test, !(colnames(Xstnd) %in% "int"), drop=FALSE ]


X_train <- Xstnd[ xndx_train, , drop=FALSE ]
X_test  <- Xstnd[ xndx_test, , drop=FALSE ]

y_train <- yy[ xndx_train ]
y_test  <- yy[ xndx_test ]



library(mactivate)

m_tot <- 14

xcmact_hybrid <-
f_control_mactivate(
param_sensitivity = 10^11,
bool_free_w       = TRUE,
w0_seed           = -0.001,
#w0_seed           = 0,
#w_col_search      = "one",
w_col_search      = "alternate",
bool_headStart    = TRUE,
antifreeze        = TRUE, #### must be true to escape
max_internal_iter = 100, #####
ss_stop           = 10^(-24), ### -14 very small
#escape_rate       = 1.001,
#escape_rate       = 1.0001,
escape_rate       = 1.0001,
step_size         = 1/300,
Wadj              = 1/300,
#force_tries       = 100,
force_tries       = 0,
lambda            = 0/300000, #### ridge on primaries
#tol               = 10^(-18) ### -14 hybrid only
tol               = 10^(-24) ### -14 hybrid only
)



xcmact_hybrid <-
f_control_mactivate(
param_sensitivity = 10^11,
bool_free_w       = TRUE,
w0_seed           = 0.000,
#w0_seed           = 0,
#w_col_search      = "one",
w_col_search      = "alternate",
bool_headStart    = TRUE,
antifreeze        = TRUE, #### must be true to escape
max_internal_iter = 100, #####
ss_stop           = 10^(-24), ### -14 very small
#escape_rate       = 1.001,
#escape_rate       = 1.0001,
escape_rate       = 1.0001,
step_size         = 1/300,
Wadj              = 1/300,
#force_tries       = 100,
force_tries       = 500,
lambda            = 1/1000, #### ridge on primaries
#tol               = 10^(-18) ### -14 hybrid only
tol               = 10^(-24) ### -14 hybrid only
)




### source( file.path(xpath_files, "Creations", "R", "_Rlibs", "mactivateRlib", "__experimental", "___f_funs.R") )

## xgreg <- c(NA, 1/1000, 1/1000)

xxnow <- Sys.time()
xxls_out <-
#f_fit_hybrid_04( greg = xgreg, ccstarts=c(-0.1, 0.1),
f_fit_hybrid_01(
#f_fit_gradient_01(
#f_fit_gradient_logistic_04( greg = xgreg, yclip = 50, ccstarts=c(-0.1, 0.1),
X = X_train,
y = y_train,
m_tot = m_tot,
U = U_train,
m_start = 1,
mact_control = xcmact_hybrid,
#mact_control = xcmact_gradient,
verbosity = 1
)
cat( difftime(Sys.time(), xxnow, units="mins"), "\n" )


###########


yhatTT <- matrix(NA, length(xndx_test), m_tot+1)

for(iimm in 0:m_tot) {
    yhat_fold <- predict(object=xxls_out, X0=X_test, U0=U_test, mcols=iimm )
    yhatTT[ , iimm + 1 ] <- yhat_fold
}

errs_by_m <- NULL
for(iimm in 1:ncol(yhatTT)) {
    yhatX <- yhatTT[ , iimm]
    errs_by_m[ iimm ] <- sqrt(mean( (y_test - yhatX)^2 ))
    cat(iimm, "::", errs_by_m[ iimm ])
}


1 - errs_by_m[ iimm ]^2 / var(y_test)


## 9.089918 -- 0.9525754 ### using second xcmact_hybrid




#######################################

xmore_than_2W_remain <- 2 * (xmx_wickets_filled[ , c("W90") ] <= 8) - 1
xmore_than_4W_remain <- 2 * (xmx_wickets_filled[ , c("W90") ] <= 6) - 1

Xall <-
cbind(
#"int"=rep(1, length(xbatted_bin)),
xbatted_bin,
xtarget_runs,
xmore_than_2W_remain,
xmore_than_4W_remain,
xmx_runs_filled[ , c("R36", "R48", "R66", "R78", "R90") ],
xmx_wickets_filled[ , c("W36", "W48", "W66", "W78", "W90") ]
)

yall <-  xmx_runs_filled[ , c("R120") ]




xmean_train <- apply(Xall[ xndx_train, ], 2, mean)
xsd_train   <- apply(Xall[ xndx_train, ], 2, sd)



Xstnd <- t( ( t(Xall) - xmean_train ) / xsd_train )

yy <- yall


########### here
U_train <- Xstnd[ xndx_train, !(colnames(Xstnd) %in% "int"), drop=FALSE ]
U_test  <- Xstnd[ xndx_test, !(colnames(Xstnd) %in% "int"), drop=FALSE ]


X_train <- Xstnd[ xndx_train, , drop=FALSE ]
X_test  <- Xstnd[ xndx_test, , drop=FALSE ]

y_train <- yy[ xndx_train ]
y_test  <- yy[ xndx_test ]





m_tot <- 14


xcmact_hybrid <-
f_control_mactivate(
param_sensitivity = 10^11,
bool_free_w       = TRUE,
w0_seed           = 0.000,
#w0_seed           = 0,
#w_col_search      = "one",
w_col_search      = "alternate",
bool_headStart    = TRUE,
antifreeze        = TRUE, #### must be true to escape
max_internal_iter = 100, #####
ss_stop           = 10^(-24), ### -14 very small
#escape_rate       = 1.001,
#escape_rate       = 1.0001,
escape_rate       = 1.0001,
step_size         = 1/300,
Wadj              = 1/300,
#force_tries       = 100,
force_tries       = 500,
lambda            = 1/100000, #### ridge on primaries
#tol               = 10^(-18) ### -14 hybrid only
tol               = 10^(-24) ### -14 hybrid only
)




### source( file.path(xpath_files, "Creations", "R", "_Rlibs", "mactivateRlib", "__experimental", "___f_funs.R") )

## xgreg <- c(NA, 1/1000, 1/1000)

xxnow <- Sys.time()
xxls_out <-
#f_fit_hybrid_04( greg = xgreg, ccstarts=c(-0.1, 0.1),
f_fit_hybrid_01(
#f_fit_gradient_01(
#f_fit_gradient_logistic_04( greg = xgreg, yclip = 50, ccstarts=c(-0.1, 0.1),
X = X_train,
y = y_train,
m_tot = m_tot,
U = U_train,
m_start = 1,
mact_control = xcmact_hybrid,
#mact_control = xcmact_gradient,
verbosity = 1
)
cat( difftime(Sys.time(), xxnow, units="mins"), "\n" )


###########


yhatTT <- matrix(NA, length(xndx_test), m_tot+1)

for(iimm in 0:m_tot) {
    yhat_fold <- predict(object=xxls_out, X0=X_test, U0=U_test, mcols=iimm )
    yhatTT[ , iimm + 1 ] <- yhat_fold
}

errs_by_m <- NULL
for(iimm in 1:ncol(yhatTT)) {
    yhatX <- yhatTT[ , iimm]
    errs_by_m[ iimm ] <- sqrt(mean( (y_test - yhatX)^2 ))
    cat(iimm, "::", errs_by_m[ iimm ])
}


1 - errs_by_m[ iimm ]^2 / var(y_test)


## 9.079612 -- 0.9529621




#######################################


xmore_than_2W_remain <- 2 * (xmx_wickets_filled[ , c("W90") ] <= 8) - 1
xmore_than_4W_remain <- 2 * (xmx_wickets_filled[ , c("W90") ] <= 6) - 1

Xall <-
cbind(
#"int"=rep(1, length(xbatted_bin)),
xbatted_bin,
xtarget_runs,
xmore_than_2W_remain,
xmore_than_4W_remain,
xmx_runs_filled[ , c("R66", "R78", "R90") ],
xmx_wickets_filled[ , c("W66", "W78", "W90") ]
)

yall <-  xmx_runs_filled[ , c("R120") ]




xmean_train <- apply(Xall[ xndx_train, ], 2, mean)
xsd_train   <- apply(Xall[ xndx_train, ], 2, sd)



Xstnd <- t( ( t(Xall) - xmean_train ) / xsd_train )

yy <- yall


########### here
U_train <- Xstnd[ xndx_train, !(colnames(Xstnd) %in% "int"), drop=FALSE ]
U_test  <- Xstnd[ xndx_test, !(colnames(Xstnd) %in% "int"), drop=FALSE ]


X_train <- Xstnd[ xndx_train, , drop=FALSE ]
X_test  <- Xstnd[ xndx_test, , drop=FALSE ]

y_train <- yy[ xndx_train ]
y_test  <- yy[ xndx_test ]





m_tot <- 14


xcmact_hybrid <-
f_control_mactivate(
param_sensitivity = 10^11,
bool_free_w       = TRUE,
w0_seed           = 0.000,
#w0_seed           = 0,
#w_col_search      = "one",
w_col_search      = "alternate",
bool_headStart    = TRUE,
antifreeze        = TRUE, #### must be true to escape
max_internal_iter = 100, #####
ss_stop           = 10^(-24), ### -14 very small
#escape_rate       = 1.001,
#escape_rate       = 1.0001,
escape_rate       = 1.0001,
step_size         = 1/300,
Wadj              = 1/300,
#force_tries       = 100,
force_tries       = 500,
lambda            = 1/100000, #### ridge on primaries
#tol               = 10^(-18) ### -14 hybrid only
tol               = 10^(-24) ### -14 hybrid only
)




### source( file.path(xpath_files, "Creations", "R", "_Rlibs", "mactivateRlib", "__experimental", "___f_funs.R") )

## xgreg <- c(NA, 1/1000, 1/1000)

xxnow <- Sys.time()
xxls_out <-
#f_fit_hybrid_04( greg = xgreg, ccstarts=c(-0.1, 0.1),
f_fit_hybrid_01(
#f_fit_gradient_01(
#f_fit_gradient_logistic_04( greg = xgreg, yclip = 50, ccstarts=c(-0.1, 0.1),
X = X_train,
y = y_train,
m_tot = m_tot,
U = U_train,
m_start = 1,
mact_control = xcmact_hybrid,
#mact_control = xcmact_gradient,
verbosity = 1
)
cat( difftime(Sys.time(), xxnow, units="mins"), "\n" )


###########


yhatTT <- matrix(NA, length(xndx_test), m_tot+1)

for(iimm in 0:m_tot) {
    yhat_fold <- predict(object=xxls_out, X0=X_test, U0=U_test, mcols=iimm )
    yhatTT[ , iimm + 1 ] <- yhat_fold
}

errs_by_m <- NULL
for(iimm in 1:ncol(yhatTT)) {
    yhatX <- yhatTT[ , iimm]
    errs_by_m[ iimm ] <- sqrt(mean( (y_test - yhatX)^2 ))
    cat(iimm, "::", errs_by_m[ iimm ])
}


1 - errs_by_m[ iimm ]^2 / var(y_test)


## 9.133944 -- 0.953346

plot( yhatTT[ , ncol(yhatTT)], y_test )

plot( Xall[ , "R90" ], yall )


#######################################
#######################################
#######################################
#######################################


xmx_r <- xmx_runs_filled[ , c("R24", "R48", "R60") ]
xmx_w <- xmx_wickets_filled[ , c("W24", "W48", "W60") ]

xmore_than_2W_remain <- 2 * (xmx_wickets_filled[ , c("W60") ] <= 6) - 1

Xall <-
cbind(
#"int"=rep(1, length(xbatted_bin)),
xbatted_bin,
xtarget_runs,
xmore_than_2W_remain,
log(xtarget_runs + 1),
xmx_r,
xmx_w,
log(xmx_r + 1),
log(xmx_w + 1)
)

yall <-  xmx_runs_filled[ , c("R120") ]

xmean_train <- apply(Xall[ xndx_train, ], 2, mean)
xsd_train   <- apply(Xall[ xndx_train, ], 2, sd)


Xstnd <- t( ( t(Xall) - xmean_train ) / xsd_train )

## Xstnd <- Xall

yy <- yall


########### here
U_train <- Xstnd[ xndx_train, !(colnames(Xstnd) %in% "int"), drop=FALSE ]
U_test  <- Xstnd[ xndx_test, !(colnames(Xstnd) %in% "int"), drop=FALSE ]


X_train <- Xstnd[ xndx_train, , drop=FALSE ]
X_test  <- Xstnd[ xndx_test, , drop=FALSE ]

y_train <- yy[ xndx_train ]
y_test  <- yy[ xndx_test ]





m_tot <- 10


xcmact_hybrid <-
f_control_mactivate(
param_sensitivity = 10^11,
bool_free_w       = TRUE,
w0_seed           = 0.000,
#w0_seed           = 0,
#w_col_search      = "one",
w_col_search      = "alternate",
bool_headStart    = TRUE,
antifreeze        = TRUE, #### must be true to escape
max_internal_iter = 100, #####
ss_stop           = 10^(-24), ### -14 very small
#escape_rate       = 1.001,
#escape_rate       = 1.0001,
escape_rate       = 1.0001,
step_size         = 1/300,
Wadj              = 1/300,
#force_tries       = 100,
force_tries       = 500,
lambda            = 1/100000, #### ridge on primaries
#tol               = 10^(-18) ### -14 hybrid only
tol               = 10^(-24) ### -14 hybrid only
)




### source( file.path(xpath_files, "Creations", "R", "_Rlibs", "mactivateRlib", "__experimental", "___f_funs.R") )

## xgreg <- c(NA, 1/1000, 1/1000)

xxnow <- Sys.time()
xxls_out <-
#f_fit_hybrid_04( greg = xgreg, ccstarts=c(-0.1, 0.1),
f_fit_hybrid_01(
#f_fit_gradient_01(
#f_fit_gradient_logistic_04( greg = xgreg, yclip = 50, ccstarts=c(-0.1, 0.1),
X = X_train,
y = y_train,
m_tot = m_tot,
U = U_train,
m_start = 1,
mact_control = xcmact_hybrid,
#mact_control = xcmact_gradient,
verbosity = 1
)
cat( difftime(Sys.time(), xxnow, units="mins"), "\n" )


###########


yhatTT <- matrix(NA, length(xndx_test), m_tot+1)

for(iimm in 0:m_tot) {
    yhat_fold <- predict(object=xxls_out, X0=X_test, U0=U_test, mcols=iimm )
    yhatTT[ , iimm + 1 ] <- yhat_fold
}

errs_by_m <- NULL
for(iimm in 1:ncol(yhatTT)) {
    yhatX <- yhatTT[ , iimm]
    errs_by_m[ iimm ] <- sqrt(mean( (y_test - yhatX)^2 ))
    cat(iimm, "::", errs_by_m[ iimm ])
}


1 - errs_by_m[ iimm ]^2 / var(y_test)


## 14.28775 -- 0.8754508


######### random forest

library(randomForest)
#library(datasets)
#library(caret)

xdfTrain <- data.frame("y"=y_train, X_train)

rf <- randomForest(y~., data=xdfTrain, proximity=TRUE) ; print(rf)

xdfTest <- data.frame("y"=y_test, X_test)

p1 <- predict(rf, xdfTest)
#confusionMatrix(p1, train$ Species)

xrf_rmse <- sqrt( sum( (y_test - p1)^2 ) / length(xndx_test) ) ; xrf_rmse

1 - xrf_rmse^2 / var(y_test)

#### 0.8667657


####### more directions


m_tot <- 14


xcmact_hybrid <-
f_control_mactivate(
param_sensitivity = 10^11,
bool_free_w       = TRUE,
w0_seed           = 0.000,
#w0_seed           = 0,
#w_col_search      = "one",
w_col_search      = "alternate",
bool_headStart    = TRUE,
antifreeze        = TRUE, #### must be true to escape
max_internal_iter = 100, #####
ss_stop           = 10^(-24), ### -14 very small
#escape_rate       = 1.001,
#escape_rate       = 1.0001,
escape_rate       = 1.0001,
step_size         = 1/30,
Wadj              = 1/3,
#force_tries       = 100,
force_tries       = 1000,
lambda            = 1/1000000, #### ridge on primaries
#tol               = 10^(-18) ### -14 hybrid only
tol               = 10^(-24) ### -14 hybrid only
)




### source( file.path(xpath_files, "Creations", "R", "_Rlibs", "mactivateRlib", "__experimental", "___f_funs.R") )

## xgreg <- c(NA, 1/1000, 1/1000)

xxnow <- Sys.time()
xxls_out <-
#f_fit_hybrid_04( greg = xgreg, ccstarts=c(-0.1, 0.1),
f_fit_hybrid_01(
#f_fit_gradient_01(
#f_fit_gradient_logistic_04( greg = xgreg, yclip = 50, ccstarts=c(-0.1, 0.1),
X = X_train,
y = y_train,
m_tot = m_tot,
U = U_train,
m_start = 1,
mact_control = xcmact_hybrid,
#mact_control = xcmact_gradient,
verbosity = 1
)
cat( difftime(Sys.time(), xxnow, units="mins"), "\n" )


###########


yhatTT <- matrix(NA, length(xndx_test), m_tot+1)

for(iimm in 0:m_tot) {
    yhat_fold <- predict(object=xxls_out, X0=X_test, U0=U_test, mcols=iimm )
    yhatTT[ , iimm + 1 ] <- yhat_fold
}

errs_by_m <- NULL
for(iimm in 1:ncol(yhatTT)) {
    yhatX <- yhatTT[ , iimm]
    errs_by_m[ iimm ] <- sqrt(mean( (y_test - yhatX)^2 ))
    cat(iimm, "::", errs_by_m[ iimm ])
}


1 - errs_by_m[ iimm ]^2 / var(y_test)


## 14.21362 -- 0.8738303

