//Example for computing trade signals in R
#include <r.h>

int Size = 200; // number of candles needed by the R algorithm  

bool RCheck()
{
   if(!Rrun()) {
     quit("R session aborted!");
     return false;     
   }
   return true;
}

function run()
{
  BarPeriod = 60;
  LookBack = Size;
  asset("EUR/USD");  
  
  if(is(INITRUN)) {
    Rstart("main.R",2);
	// load all required R objects from a file in the Zorro Data folder 
    Rx(strf("load('%sData/MyObjects.bin')",slash(ZorroFolder)));
	// make sure everything loaded ok
    if(!RCheck()) return;
	}

	// generate reverse price series (latest price comes last)
	  vars O = rev(series(priceOpen())),
		H = rev(series(priceHigh())),
		L = rev(series(priceLow())),
		C = rev(series(priceClose()));
		
	  if(!is(LOOKBACK)) {
	// send the last 200 candles to R
		Rset("Open",O,Size);
		Rset("High",H,Size);
		Rset("Low",L,Size);
		Rset("Close",C,Size);

	// let R compute the signal
		var Signal = Rd("Compute(Open,High,Low,Close)");
		if(!RCheck()) return;
		
	// test a function
		Rx("RSum = function(MyVector) { return(sum(MyVector)) }");
		var RSum = Rd("RSum(Rout)"); 
		printf("\nSum: %.0f",RSum);
	}
}