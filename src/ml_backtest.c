#include <r.h>

function main()
{
	if(!Rstart("", 2)) {	//enable verbose output
		printf("Error - R won't start!");
		return;
	}
	//Rx("rm(list = ls());"); // clear the workspace
	
	if(!is(LOOKBACK)) {		
		Rset("Rin",vecIn,5);
		Rx("");
		Rv("Rout",vecOut,5);
	}
	
	if(!Rrun()) {
		printf("Error - R session aborted!");
		return;
	} 
	
	
// test a function
	Rx("RSum = function(MyVector) { return(sum(MyVector)) }");
	var RSum = Rd("RSum(Rout)"); 
	printf("\nSum: %.0f",RSum);
}
