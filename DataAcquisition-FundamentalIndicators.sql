/******
Created on Mon Aug 17 2020
Author: Brandi Beals
Description: Thesis Data Acquisition
******/

-- Let's begin by examining the benchmarks
SELECT TOP (1000) *
  FROM [ArtisanApplication].[edm].[Index]
  WHERE name IN ('Russell 3000 Index With Dividends', 'Russell 2000 Index With Dividends', 'Russell 1000 Index With Dividends', 'Russell Midcap Growth Index With Dividends')

-- Then we need to know what accounts we're working with
SELECT TOP (1000) *
  FROM [ArtisanApplication].[edm].[Account]
  WHERE InvestmentTeamId='6' AND vehicleTypeId='2'

-- Next, we'll get our universe of securities
SELECT TOP (10000)
	EDMSecurityId,
	IssueCurrencyCode,
	IndexName,
	ArtisanId,
	Ticker,
	TickerAndExchange,
	SecurityDescription,
	SecurityType,
	AssetType,
	GICSSector,
	GICSIndustry
  FROM [ArtisanWarehouseReporting].[cube].[IndexConstituent]
  LEFT JOIN [ArtisanWarehouseReporting].cube.BenchmarkIndex ON BenchmarkIndex.IndexKey = IndexConstituent.IndexKey
  LEFT JOIN [ArtisanWarehouseReporting].cube.Security ON Security.SecurityKey = IndexConstituent.SecurityKey
  WHERE IndexName IN ('Russell 2000 Index With Dividends', 'Russell 1000 Index With Dividends')
  AND EffectiveDate='08-31-2020' AND SecurityType='COM'

-- Then look up the closing price
SELECT TOP (1000)
	edmSecurityId,
	effectiveDate,
	priceLocalAdjusted,
	priceLocalUnadjusted,
	volume,
	vwapLocal
FROM [ArtisanApplication].[edm].[SecurityPriceBloombergEod]
WHERE effectiveDate>='01-01-2020'

-- Get various prices and returns
SELECT TOP (1000)
	securityId,
	effectiveDate,
	netChg1d,
	pctChg1d,
	localPrice,
	localPriceOpen,
	localPriceClose,
	localHigh52week,
	localLow52week,
	currMktCap,
	currMktCapUSD,
	netDebt
FROM [ArtisanRepository].[dbo].[aplp_bloomberg_time_series_data]
LEFT JOIN [ArtisanRepository].[dbo].[aplp_instrument] ON aplp_instrument.instrumentId = aplp_bloomberg_time_series_data.instrumentId
WHERE effectiveDate>='01-01-2020'
AND securityId='70001920'
ORDER BY effectiveDate DESC

-- Get additional features
SELECT TOP (1000)
	securityId,
	effectiveDate,
	marketCap,
	annualDividend,
	dividendYield,
	volume,
	vwap,
	pbRatio,
	eps,
	peFy1,
	bvps,
	roe,
	fiscalFreeCashFlowEstimate,
	cash,
	ebit,
	ebitda,
	longTermDebt,
	netIncome,
	operatingExpense,
	totalAssets,
	totalDebt,
	totalEquity,
	totalLiabilities,
	totalFeeIncome,
	--betaSP500,
	betaACWI
	--price3mAgo,
	--localPrice1yrAgo,
	--localPrice3yrAgo,
	--priceChg3M,
	--priceChg52Wk,
	--totalReturn3m,
	--totalReturn1y
FROM [ArtisanRepository].[dbo].[aplp_factset_time_series_data]
LEFT JOIN [ArtisanRepository].[dbo].[aplp_instrument] ON aplp_instrument.instrumentId = aplp_factset_time_series_data.instrumentId
WHERE effectiveDate>='01-01-2020'
AND securityId='70001920'
ORDER BY effectiveDate DESC

-- Get 3 month performance and 3 year performance
SELECT TOP (1000)
	edmSecurityId,
	effectiveDate,
	totalReturn3MonthLocal,
	totalReturn3YearLocal
FROM [ArtisanApplication].[edm].[SecurityReturnFactSetEod]
WHERE effectiveDate>='01-01-2020'

-- IP
SELECT TOP (10000)
	edmSecurityId,
	effectiveDate,
	rawX,
	rawY,
	rawZ,
	name AS formula
FROM [ArtisanApplication].[grwimp].[MarketMapQuartileInputs]
LEFT JOIN grwimp.MarketMapFormula ON MarketMapFormula.id = MarketMapQuartileInputs.formulaId
WHERE effectiveDate>='01-01-2020'



-- Oh, and we'll need our dependent variable, returns
--ArtisanApplication.edm.SecurityReturnBloombergEod
--pricePct1DLocal
--totalReturnGrossPct1DLocal


/******
Unused queries
******/

-- Another approach to security reference data
SELECT
ConstituentList.edmSecurityId,
InvestmentTeamName,
InvestmentTeamCode,
ConstituentListType.name AS ConstituentListType,
SecurityReference.name AS SecurityName,
AssetType.name AS AssetType,
SecurityType.name AS SecurityType,
SecurityType.code AS SecurityTypeCode,
PrimaryExchange.name AS PrimaryExchange,
PrimaryExchange.code AS PrimaryExchangeCode,
HomogenousGroup.name AS HomogenousGroup,
FoodChain.name AS FoodChain

--artisanSecurityId,
--issueCountryId,
--riskCountryId,
--edmIssuerId,
--tradeCurrencyId,
--AssetCurrency,

FROM edm.ConstituentList
LEFT JOIN edm.ConstituentListType ON ConstituentListType.id = ConstituentList.constituentListTypeId
LEFT JOIN edm.InvestmentTeam ON InvestmentTeam.id = ConstituentList.investmentTeamId
LEFT JOIN edm.SecurityReference ON SecurityReference.edmSecurityId = ConstituentList.edmSecurityId
LEFT JOIN edm.AssetType ON AssetType.id = SecurityReference.assetTypeId
LEFT JOIN edm.SecurityType ON SecurityType.id = SecurityReference.securityTypeId
--LEFT JOIN edm.CompositeExchange ON CompositeExchange.id = SecurityReference.compositeExchangeId
LEFT JOIN edm.PrimaryExchange ON PrimaryExchange.id = SecurityReference.primaryExchangeId
LEFT JOIN grwimp.IssuerHomogenousGroup ON IssuerHomogenousGroup.edmIssuerId = SecurityReference.edmIssuerId
LEFT JOIN grwimp.HomogenousGroup ON HomogenousGroup.id = IssuerHomogenousGroup.homogenousGroupId
LEFT JOIN grwimp.FoodChain ON FoodChain.id = HomogenousGroup.foodChainId

WHERE InvestmentTeamCode='GRW'
AND	SecurityType.code='COM'

-- This table has the price for securities held by portfolios
SELECT DISTINCT
	--edmAccountId,
	edmSecurityId,
	effectiveDate,
	priceUsd
FROM [ArtisanApplication].[edm].[AccountPositionEod]
WHERE longShortInd='L'
AND effectiveDate>='01-01-2020'
--AND edmAccountId IN ('50000195','50000333','50000373','50097342')

-- And the opening price (this is only current day)
SELECT DISTINCT
	--edmAccountId,
	edmSecurityId,
	effectiveDate,
	priceUsd
FROM [ArtisanApplication].[edm].[AccountPositionBod]
WHERE longShortInd='L'
AND effectiveDate>='01-01-2020'
--AND edmAccountId IN ('50000195','50000333','50000373','50097342')

-- This table gives the weights each index has in each security
SELECT
	effectiveDate,
	shares,
	weight,
	[Index].name AS indexName,
	SecurityReference.name AS securityName,
	artisanSecurityId,	
	assetTypeId,
	securityTypeId,
	issueCountryId,
	tradeCurrencyId,
	AssetCurrency
  FROM [ArtisanApplication].[edm].IndexPositionEod
  LEFT JOIN edm.[Index] ON [Index].edmIndexId = IndexPositionEod.edmIndexId
  LEFT JOIN edm.SecurityReference ON SecurityReference.edmSecurityId = IndexPositionEod.edmSecurityId
  WHERE [Index].name IN ('Russell 2000 Index With Dividends') --MSCI All Country World Index (Net)') -- ('Russell Midcap Growth Index With Dividends', 'Russell 2000 Index With Dividends')
  AND effectiveDate='7-1-2020'
  ORDER BY artisanSecurityId

-- Data for report that analyzes closed positions
SELECT TOP 10000 *
FROM [ArtisanApplication].[grwimp].[ClosedPositionReportRecord]

SELECT TOP 10000 *
FROM [ArtisanApplication].[grwimp].[AccountPositionCloseReturn]

-- Additional features like market cap
SELECT TOP (1000)
	edmSecurityId,
	effectiveDate,
	--exchangeTypeId,
	priceHigh52WeekLocal,
	priceLow52WeekLocal,
	volume10DAverage,
	volume3MAverage,
	dailyValueTraded20DAverageLocal,
	dailyValueTraded3MAverageLocal,
	currentMktCapMillionsUsd
FROM [ArtisanApplication].[edm].[SecurityPriceMetricsBloomberg]
WHERE effectiveDate>='01-01-2020'