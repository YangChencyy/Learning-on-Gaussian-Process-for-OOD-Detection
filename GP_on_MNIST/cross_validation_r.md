```r
LEAVE10OUTPRED = rep(0,n)
	for(k in 1:10){
	  LEAVEOUTINDS = k+10*(0:(floor(n / 10)))
	  LEAVEOUTINDS = LEAVEOUTINDS[LEAVEOUTINDS <= n]
	  REMAININGINDS = setdiff(1:n, LEAVEOUTINDS)
	  Xtrain = w_X[REMAININGINDS,]
	  Ytrain = w_Y[REMAININGINDS]
	  Xpred = w_X[LEAVEOUTINDS,]	  
	  GP <- GPfitting(Xtrain,Ytrain)
	  PredL10O <- GPpred(GP, Xpred)
	  LEAVE10OUTPRED[LEAVEOUTINDS] = PredL10O$pred
	}

```

