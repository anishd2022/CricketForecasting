
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

