********************************************************************************
****************************** REGRESSION DO FILE ******************************
********************************************************************************
** Deaton, B. James and B. Lipka. 2021. The Provision of drinking water in *****
** First Nations communities and Ontario municipalities: Insight into the ******
** emergence of water sharing arrangements. Ecological Economics, 198(107147) **
** https://doi.org/10.1016/j.ecolecon.2021.107147 ******************************
********************************************************************************


*************
** TABLE 1 **
*************

* Ontario
su WSA North PD Dist Inc Elev if WSA != . & PD != .
* First Nations Communities
su WSA North PD Dist Inc Elev if FN == 1 & WSA != . & PD != .
* Municipalities
su WSA North PD Dist Inc Elev if FN == 0 & WSA != . & PD != .



*************
** TABLE 2 **
*************

*************** Ontario ***************

* Model 1: OLS - FN variable only 
regress WSA i.FN, robust
* Summary statistics for regression sample
estat su

* Model 2: OLS - full suite of covariates
regress WSA i.FN i.North lnPD lnDist lnInc Ele, robust 	
* Summary statistics for regression sample
estat su

********** Northern Ontario ************

* Model 1: OLS - FN variable only 
regress WSA i.FN if North == 1, robust
* Summary statistics for regression sample
estat su

* Model 2: OLS - full suite of covariates
regress WSA i.FN lnPD lnDist lnInc Ele if North == 1, robust
* Summary statistics for regression sample
estat su

********** Southern Ontario *************

* Model 1: OLS - FN variable only 
regress WSA i.FN if North == 0, robust
* Summary statistics for regression sample
estat su

* Model 2: OLS - full suite of covariates 
regress WSA i.FN lnPD lnDist lnInc Ele if North == 0, robust
* Summary statistics for regression sample
estat su



*************
** TABLE 3 **
************

* Model 2: OLS with standardized coefficients - Ontario, full suite of covariates
regress WSA i.FN i.North lnPD lnDist lnInc Ele, robust beta		
* Summary statistics for regression sample
estat su



*****************************
** TABLE A4.1 (Appendix 4) **
*****************************

* Model 2: probit - Ontario, full suite of covariates
probit WSA i.FN i.North lnPD lnDist lnInc Ele, robust	
* Average marginal effects
margins, dydx(i.FN i.North lnPD lnDist lnInc Ele) post
* Summary statistics for regression sample
estat su

* Model 2: probit - north Ontario, full suite of covariates
probit WSA i.FN lnPD lnDist lnInc Ele if North == 1, robust
* Average marginal effects
margins, dydx(i.FN lnPD lnDist lnInc Ele) post
*Summary statistics for regression sample
estat su

* Model 2: probit - south Ontario, full suite of covariates
probit WSA i.FN lnPD lnDist lnInc Ele if North == 0, robust
* Average marginal effects
margins, dydx(i.FN lnPD lnDist lnInc Ele) post
* Summary statistics for regression sample
estat su



****************
** TABLE A4.2 **
****************

* Model 2: OLS with untransformed variables - Ontario, full suite of covariates 
regress WSA i.FN i.North PD Dist Inc Ele, robust
* Summary statistics for regression sample
estat su

* Model 2: OLS with untransformed variables - north Ontario, full suite of covariates
regress WSA i.FN PD Dist Inc Ele if North == 1, robust
* Summary statistics for regression sample
estat su

* Model 2: OLS with untransformed variables - south Ontario, full suite of covariates
regress WSA i.FN PD Dist Inc Ele if North == 0, robust
* Summary statistics for regression sample
estat su



*******
**END**
*******
