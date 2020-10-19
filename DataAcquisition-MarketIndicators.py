# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 2020
Author: Brandi Beals
Description: Thesis Data Acquisition
"""

######################################
# IMPORT PACKAGES
######################################

from datetime import datetime
import pandas as pd
import pandas_datareader.data as web
import pandas_datareader.iex as iex
import os

######################################
# DEFIINITIONS
######################################

os.chdir(r'C:\Users\bbeals\Dropbox (Personal)\Masters in Predictive Analytics\590-Thesis\Data')
now_time = datetime.now()
start_time = datetime(2018, 11 , 1)

######################################
# GET DATA
######################################

# Federal Reserve Economic Data (FRED)
# https://fred.stlouisfed.org/
# found this blog helpful: https://medium.com/swlh/pandas-datareader-federal-reserve-economic-data-fred-a360c5795013

# https://fred.stlouisfed.org/series/DFF
FREDinterestrate = web.DataReader('DFF', 'fred', start_time, now_time)
# https://fred.stlouisfed.org/series/DEXUSEU
FREDexchangerate = web.DataReader('DEXUSEU', 'fred', start_time, now_time)

# Fama-French Data
# http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

# http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html
famafrench = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench')
famafrench = famafrench[0]
# http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor_daily.html
famafrenchMom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench')
famafrenchMom = famafrenchMom[0]

######################################
# COMPILE DATA FILE
######################################

output_df = FREDinterestrate.join(FREDexchangerate)
output_df = output_df.join(famafrench)
output_df = output_df.join(famafrenchMom)

output_df.to_csv(r'Market Indicators.csv')





######################################
# WORK IN PROGRESS
######################################

universe = pd.read_csv('Universe.csv')
symbols = universe['Ticker'].to_list()
bad_symbols = []

api_key = 'Tpk_c5b53b4d0ea4410fba7dfaabc9d93aa5'
data = pd.DataFrame(columns = ['open','high','low','close','volume','name'])
os.environ['IEX_SANDBOX'] = 'enable' # https://sandbox.iexapis.com/ 

df = web.DataReader('dataset=FREDVOLIND&v=Title&h=TIME&from=2018-12-09&to=2019-01-09', 'econdb')
#web.DataReader('ticker=RGDPUS', 'econdb')

# IEX
# https://iexcloud.io/cloud-login
for s in symbols:
    try:
        iex = web.DataReader(s, 'iex', start=start_time, end=now_time, api_key=api_key)
        iex['name'] = s
        data = pd.concat([data, iex])
    except:
        bad_symbols.append(s)

data.reset_index(level=0, inplace=True)
data.rename(columns = {'index':'date'}, inplace=True)



#df['MA20'] = df['Adj Close'].rolling(window=20).mean()
#df['MA60'] = df['Adj Close'].rolling(window=60).mean()



test_df = web.get_data_yahoo('FB', start_time, now_time)





"""
symbols = ['A', 'AA', 'AAL', 'AAN', 'AAOI', 'AAON', 'AAP', 'AAPL', 'AAWW', 'AAXN', 'ABBV',
'ABC', 'ABCB', 'ABEO', 'ABG', 'ABM', 'ABMD', 'ABT', 'ABTX', 'AC', 'ACA',
'ACAD', 'ACBI', 'ACCO', 'ACEL', 'ACGL', 'ACHC', 'ACIA', 'ACIW', 'ACLS', 'ACM',
'ACN', 'ACNB', 'ACRX', 'ACTG', 'ADBE', 'ADES', 'ADI', 'ADM', 'ADMA', 'ADNT',
'ADP', 'ADPT', 'ADRO', 'ADS', 'ADSK', 'ADSW', 'ADT', 'ADTN', 'ADUS', 'ADVM',
'AE', 'AEE', 'AEGN', 'AEIS', 'AEL', 'AEO', 'AEP', 'AERI', 'AES', 'AFG', 'AFL',
'AFMD', 'AGCO', 'AGEN', 'AGFS', 'AGIO', 'AGLE', 'AGM', 'AGO', 'AGR', 'AGRX',
'AGS', 'AGTC', 'AGX', 'AGYS', 'AHCO', 'AIG', 'AIMC', 'AIMT', 'AIN', 'AIR',
'AIT', 'AIZ', 'AJG', 'AJRD', 'AKAM', 'AKBA', 'AKCA', 'AKRO', 'AKTS', 'AL',
'ALB', 'ALBO', 'ALCO', 'ALE', 'ALEC', 'ALG', 'ALGN', 'ALGT', 'ALK', 'ALKS',
'ALL', 'ALLE', 'ALLK', 'ALLO', 'ALLY', 'ALNY', 'ALRM', 'ALRS', 'ALSK', 'ALSN',
'ALTA', 'ALTG', 'ALTR', 'ALXN', 'AM', 'AMAG', 'AMAL', 'AMAT', 'AMBA', 'AMBC',
'AMC', 'AMCR', 'AMCX', 'AMD', 'AME', 'AMED', 'AMEH', 'AMG', 'AMGN', 'AMK',
'AMKR', 'AMN', 'AMNB', 'AMOT', 'AMP', 'AMPH', 'AMRC', 'AMRK', 'AMRS', 'AMRX',
'AMSC', 'AMSF', 'AMSWA', 'AMTB', 'AMTD', 'AMWD', 'AMZN', 'AN', 'ANAB', 'ANAT',
'ANDE', 'ANET', 'ANF', 'ANGO', 'ANIK', 'ANIP', 'ANSS', 'ANTM', 'AON', 'AOS',
'AOSL', 'APA', 'APAM', 'APD', 'APEI', 'APG', 'APH', 'APLS', 'APLT', 'APO',
'APOG', 'APPF', 'APPN', 'APPS', 'APRE', 'APT', 'APTV', 'APTX', 'APYX', 'AQST',
'AQUA', 'AR', 'ARA', 'ARAV', 'ARAY', 'ARCB', 'ARCH', 'ARCT', 'ARD', 'ARDX',
'ARES', 'ARGO', 'ARL', 'ARLO', 'ARMK', 'ARNA', 'ARNC', 'AROC', 'AROW', 'ARQT',
'ARTNA', 'ARVN', 'ARW', 'ARWR', 'ASB', 'ASC', 'ASGN', 'ASH', 'ASIX', 'ASMB',
'ASPN', 'ASPS', 'ASPU', 'ASTE', 'ASUR', 'AT', 'ATEC', 'ATEN', 'ATEX', 'ATGE',
'ATH', 'ATHX', 'ATI', 'ATKR', 'ATLC', 'ATLO', 'ATNI', 'ATNX', 'ATO', 'ATOM',
'ATR', 'ATRA', 'ATRC', 'ATRI', 'ATRO', 'ATRS', 'ATSG', 'ATUS', 'ATVI', 'ATXI',
'AUB', 'AUBN', 'AVA', 'AVAV', 'AVCO', 'AVD', 'AVEO', 'AVGO', 'AVID', 'AVLR',
'AVNS', 'AVNT', 'AVRO', 'AVT', 'AVTR', 'AVXL', 'AVY', 'AVYA', 'AWH', 'AWI',
'AWK', 'AWR', 'AX', 'AXDX', 'AXGN', 'AXL', 'AXLA', 'AXNX', 'AXP', 'AXS',
'AXSM', 'AXTA', 'AXTI', 'AYI', 'AYTU', 'AYX', 'AZO', 'AZPN', 'AZZ', 'B', 'BA',
'BAC', 'BAH', 'BANC', 'BAND', 'BANF', 'BANR', 'BAX', 'BBBY', 'BBCP', 'BBIO',
'BBSI', 'BBX', 'BBY', 'BC', 'BCBP', 'BCC', 'BCEI', 'BCEL', 'BCLI', 'BCML',
'BCO', 'BCOR', 'BCOV', 'BCPC', 'BCRX', 'BDC', 'BDGE', 'BDSI', 'BDTX', 'BDX',
'BE', 'BEAM', 'BEAT', 'BECN', 'BELFB', 'BEN', 'BEPC', 'BERY', 'BF/A', 'BF/B',
'BFAM', 'BFC', 'BFIN', 'BFST', 'BFYT', 'BG', 'BGCP', 'BGS', 'BGSF', 'BH/A',
'BH', 'BHB', 'BHE', 'BHF', 'BHLB', 'BHVN', 'BIG', 'BIIB', 'BILL', 'BIO',
'BIPC', 'BJ', 'BJRI', 'BK', 'BKD', 'BKE', 'BKH', 'BKI', 'BKNG', 'BKR', 'BKU',
'BL', 'BLBD', 'BLD', 'BLDR', 'BLFS', 'BLK', 'BLKB', 'BLL', 'BLMN', 'BLPH',
'BLUE', 'BLX', 'BMCH', 'BMI', 'BMRC', 'BMRN', 'BMTC', 'BMY', 'BNFT', 'BOCH',
'BOH', 'BOKF', 'BOMN', 'BOOM', 'BOOT', 'BOX', 'BPFH', 'BPMC', 'BPOP', 'BPRN',
'BR', 'BRBR', 'BRC', 'BREW', 'BRID', 'BRK/B', 'BRKL', 'BRKR', 'BRKS', 'BRO',
'BRP', 'BRY', 'BSBK', 'BSGM', 'BSIG', 'BSRR', 'BSTC', 'BSVN', 'BSX', 'BTAI',
'BTU', 'BURL', 'BUSE', 'BV', 'BWA', 'BWB', 'BWFG', 'BWXT', 'BXG', 'BXS', 'BY',
'BYD', 'BYND', 'BYSI', 'BZH', 'C', 'CABA', 'CABO', 'CAC', 'CACC', 'CACI',
'CADE', 'CAG', 'CAH', 'CAI', 'CAKE', 'CAL', 'CALA', 'CALB', 'CALM', 'CALX',
'CAMP', 'CAR', 'CARA', 'CARE', 'CARG', 'CARR', 'CARS', 'CASA', 'CASH', 'CASI',
'CASS', 'CASY', 'CAT', 'CATB', 'CATC', 'CATM', 'CATO', 'CATY', 'CB', 'CBAN',
'CBAY', 'CBB', 'CBFV', 'CBIO', 'CBMG', 'CBNK', 'CBOE', 'CBRE', 'CBRL', 'CBSH',
'CBT', 'CBTX', 'CBU', 'CBZ', 'CC', 'CCB', 'CCBG', 'CCF', 'CCK', 'CCL', 'CCMP',
'CCNE', 'CCOI', 'CCRN', 'CCS', 'CCXI', 'CDAY', 'CDE', 'CDK', 'CDLX', 'CDMO',
'CDNA', 'CDNS', 'CDTX', 'CDW', 'CDXC', 'CDXS', 'CDZI', 'CE', 'CECE', 'CEIX',
'CELH', 'CEMI', 'CENT', 'CENTA', 'CENX', 'CERC', 'CERN', 'CERS', 'CETV',
'CEVA', 'CF', 'CFB', 'CFFI', 'CFFN', 'CFG', 'CFR', 'CFRX', 'CFX', 'CG', 'CGNX',
'CHCO', 'CHD', 'CHDN', 'CHE', 'CHEF', 'CHGG', 'CHH', 'CHMA', 'CHMG', 'CHNG',
'CHRS', 'CHRW', 'CHS', 'CHTR', 'CHUY', 'CHX', 'CI', 'CIA', 'CIEN', 'CINF',
'CIR', 'CIT', 'CIVB', 'CIX', 'CIZN', 'CKH', 'CKPT', 'CL', 'CLAR', 'CLBK',
'CLCT', 'CLDR', 'CLF', 'CLFD', 'CLGX', 'CLH', 'CLNE', 'CLR', 'CLVS', 'CLW',
'CLX', 'CLXT', 'CMA', 'CMBM', 'CMC', 'CMCL', 'CMCO', 'CMCSA', 'CMD', 'CME',
'CMG', 'CMI', 'CMP', 'CMPR', 'CMRE', 'CMRX', 'CMS', 'CMTL', 'CNA', 'CNBKA',
'CNC', 'CNCE', 'CNDT', 'CNK', 'CNMD', 'CNNE', 'CNO', 'CNOB', 'CNP', 'CNR',
'CNS', 'CNSL', 'CNST', 'CNTG', 'CNTY', 'CNX', 'CNXN', 'CODX', 'COF', 'COFS',
'COG', 'COHR', 'COHU', 'COKE', 'COLB', 'COLL', 'COLM', 'COMM', 'CONN', 'COO',
'COOP', 'COP', 'CORE', 'CORT', 'COST', 'COTY', 'COUP', 'COWN', 'CPA', 'CPB',
'CPF', 'CPK', 'CPRI', 'CPRT', 'CPRX', 'CPS', 'CPSI', 'CR', 'CRAI', 'CRBP',
'CRD/A', 'CREE', 'CRI', 'CRK', 'CRL', 'CRM', 'CRMD', 'CRMT', 'CRNC', 'CRNX',
'CROX', 'CRS', 'CRTX', 'CRUS', 'CRVL', 'CRWD', 'CRY', 'CSBR', 'CSCO', 'CSGP',
'CSGS', 'CSII', 'CSL', 'CSOD', 'CSPR', 'CSTE', 'CSTL', 'CSTR', 'CSV', 'CSWI',
'CSX', 'CTAS', 'CTB', 'CTBI', 'CTL', 'CTLT', 'CTMX', 'CTO', 'CTRN', 'CTS',
'CTSH', 'CTSO', 'CTVA', 'CTXS', 'CUB', 'CUBI', 'CUE', 'CURO', 'CUTR', 'CVA',
'CVBF', 'CVCO', 'CVCY', 'CVET', 'CVGW', 'CVI', 'CVLG', 'CVLT', 'CVLY', 'CVM',
'CVNA', 'CVS', 'CVX', 'CW', 'CWBR', 'CWCO', 'CWEN/A', 'CWEN', 'CWH', 'CWK',
'CWST', 'CWT', 'CXO', 'CYBE', 'CYCN', 'CYH', 'CYRX', 'CYTK', 'CZNC', 'CZR',
'D', 'DAKT', 'DAL', 'DAN', 'DAR', 'DBD', 'DBI', 'DBX', 'DCI', 'DCO', 'DCOM',
'DCPH', 'DD', 'DDD', 'DDOG', 'DDS', 'DE', 'DECK', 'DELL', 'DENN', 'DFIN',
'DFS', 'DG', 'DGICA', 'DGII', 'DGX', 'DHI', 'DHIL', 'DHR', 'DHT', 'DHX', 'DIN',
'DIOD', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DJCO', 'DK', 'DKS', 'DLB', 'DLTH',
'DLTR', 'DLX', 'DMRC', 'DMTK', 'DNKN', 'DNLI', 'DNOW', 'DOCU', 'DOMO', 'DOOR',
'DORM', 'DOV', 'DOW', 'DOX', 'DPZ', 'DRI', 'DRNA', 'DRQ', 'DRRX', 'DSKE',
'DSPG', 'DSSI', 'DT', 'DTE', 'DTIL', 'DUK', 'DVA', 'DVAX', 'DVN', 'DXC',
'DXCM', 'DXPE', 'DY', 'DYAI', 'DZSI', 'EA', 'EAF', 'EAT', 'EB', 'EBAY', 'EBF',
'EBIX', 'EBMT', 'EBS', 'EBSB', 'EBTC', 'ECHO', 'ECL', 'ECOL', 'ECOM', 'ECPG',
'ED', 'EDIT', 'EEFT', 'EEX', 'EFSC', 'EFX', 'EGAN', 'EGBN', 'EGHT', 'EGLE',
'EGOV', 'EGRX', 'EHC', 'EHTH', 'EIDX', 'EIG', 'EIGI', 'EIGR', 'EIX', 'EL',
'ELA', 'ELAN', 'ELF', 'ELMD', 'ELOX', 'ELY', 'EME', 'EML', 'EMN', 'EMR',
'ENDP', 'ENOB', 'ENPH', 'ENR', 'ENS', 'ENSG', 'ENTA', 'ENTG', 'ENV', 'ENVA',
'ENZ', 'EOG', 'EOLS', 'EPAC', 'EPAM', 'EPAY', 'EPC', 'EPM', 'EPZM', 'EQBK',
'EQH', 'EQT', 'ERIE', 'ERII', 'EROS', 'ES', 'ESCA', 'ESE', 'ESGR', 'ESI',
'ESNT', 'ESPR', 'ESQ', 'ESSA', 'ESTC', 'ESTE', 'ESXB', 'ETFC', 'ETH', 'ETM',
'ETN', 'ETNB', 'ETON', 'ETR', 'ETRN', 'ETSY', 'EV', 'EVBG', 'EVBN', 'EVC',
'EVER', 'EVFM', 'EVH', 'EVI', 'EVLO', 'EVOP', 'EVR', 'EVRG', 'EVRI', 'EVTC',
'EW', 'EWBC', 'EXAS', 'EXC', 'EXEL', 'EXLS', 'EXP', 'EXPD', 'EXPE', 'EXPI',
'EXPO', 'EXPR', 'EXTN', 'EXTR', 'EYE', 'EZPW', 'F', 'FAF', 'FANG', 'FARM',
'FARO', 'FAST', 'FATE', 'FB', 'FBC', 'FBHS', 'FBIO', 'FBIZ', 'FBK', 'FBM',
'FBMS', 'FBNC', 'FBP', 'FC', 'FCAP', 'FCBC', 'FCBP', 'FCCO', 'FCCY', 'FCEL',
'FCF', 'FCFS', 'FCN', 'FCNCA', 'FCX', 'FDBC', 'FDP', 'FDS', 'FDX', 'FE',
'FELE', 'FENC', 'FEYE', 'FF', 'FFBC', 'FFG', 'FFIC', 'FFIN', 'FFIV', 'FFWM',
'FGBI', 'FGEN', 'FHB', 'FHI', 'FHN', 'FI', 'FIBK', 'FICO', 'FIS', 'FISI',
'FISV', 'FIT', 'FITB', 'FIVE', 'FIVN', 'FIX', 'FIXX', 'FIZZ', 'FL', 'FLDM',
'FLGT', 'FLIC', 'FLIR', 'FLMN', 'FLNT', 'FLO', 'FLOW', 'FLR', 'FLS', 'FLT',
'FLWS', 'FLXN', 'FMAO', 'FMBH', 'FMBI', 'FMC', 'FMNB', 'FN', 'FNB', 'FNCB',
'FND', 'FNF', 'FNHC', 'FNKO', 'FNLC', 'FNWB', 'FOCS', 'FOE', 'FOLD', 'FONR',
'FOR', 'FORM', 'FORR', 'FOSL', 'FOX', 'FOXA', 'FOXF', 'FPRX', 'FRAF', 'FRBA',
'FRBK', 'FRC', 'FREQ', 'FRG', 'FRGI', 'FRME', 'FRO', 'FRPH', 'FRPT', 'FRTA',
'FSBW', 'FSFG', 'FSLR', 'FSLY', 'FSS', 'FSTR', 'FTDR', 'FTNT', 'FTV', 'FUL',
'FULC', 'FULT', 'FUNC', 'FVCB', 'FVE', 'FWRD', 'G', 'GABC', 'GAIA', 'GALT',
'GAN', 'GATX', 'GBCI', 'GBL', 'GBLI', 'GBT', 'GBX', 'GCBC', 'GCI', 'GCO',
'GCP', 'GD', 'GDDY', 'GDEN', 'GDOT', 'GDP', 'GDYN', 'GE', 'GEF/B', 'GEF',
'GENC', 'GERN', 'GES', 'GFF', 'GFN', 'GGG', 'GH', 'GHC', 'GHL', 'GHM', 'GIII',
'GILD', 'GIS', 'GKOS', 'GL', 'GLDD', 'GLIBA', 'GLNG', 'GLOB', 'GLRE', 'GLT',
'GLUU', 'GLW', 'GLYC', 'GM', 'GME', 'GMED', 'GMS', 'GNE', 'GNK', 'GNLN',
'GNMK', 'GNPX', 'GNRC', 'GNSS', 'GNTX', 'GNTY', 'GNW', 'GO', 'GOGO', 'GOLF',
'GOOG', 'GOOGL', 'GORO', 'GOSS', 'GPC', 'GPI', 'GPK', 'GPN', 'GPOR', 'GPRE',
'GPRO', 'GPS', 'GPX', 'GRA', 'GRBK', 'GRC', 'GRIF', 'GRMN', 'GRPN', 'GRTS',
'GRTX', 'GRUB', 'GRWG', 'GS', 'GSB', 'GSBC', 'GSHD', 'GSIT', 'GSKY', 'GT',
'GTES', 'GTHX', 'GTLS', 'GTN', 'GTS', 'GTT', 'GTYH', 'GVA', 'GWB', 'GWGH',
'GWRE', 'GWRS', 'GWW', 'H', 'HA', 'HAE', 'HAFC', 'HAIN', 'HAL', 'HALO', 'HARP',
'HAS', 'HAYN', 'HBAN', 'HBB', 'HBCP', 'HBI', 'HBIO', 'HBMD', 'HBNC', 'HBT',
'HCA', 'HCAT', 'HCC', 'HCCI', 'HCHC', 'HCI', 'HCKT', 'HCSG', 'HD', 'HDS', 'HE',
'HEAR', 'HEES', 'HEI/A', 'HEI', 'HELE', 'HES', 'HFC', 'HFFG', 'HFWA', 'HGV',
'HHC', 'HI', 'HIBB', 'HIFS', 'HIG', 'HII', 'HL', 'HLF', 'HLI', 'HLIO', 'HLIT',
'HLNE', 'HLT', 'HLX', 'HMHC', 'HMN', 'HMST', 'HMSY', 'HMTV', 'HNGR', 'HNI',
'HOFT', 'HOG', 'HOLX', 'HOMB', 'HOME', 'HON', 'HONE', 'HOOK', 'HOPE', 'HP',
'HPE', 'HPQ', 'HQY', 'HRB', 'HRC', 'HRI', 'HRL', 'HROW', 'HRTG', 'HRTX', 'HSC',
'HSIC', 'HSII', 'HSKA', 'HSTM', 'HSY', 'HTBI', 'HTBK', 'HTH', 'HTLD', 'HTLF',
'HTZ', 'HUBB', 'HUBG', 'HUBS', 'HUD', 'HUM', 'HUN', 'HURC', 'HURN', 'HVT',
'HWBK', 'HWC', 'HWKN', 'HWM', 'HXL', 'HY', 'HZNP', 'HZO', 'IAA', 'IAC', 'IART',
'IBCP', 'IBIO', 'IBKR', 'IBM', 'IBOC', 'IBP', 'IBTX', 'ICAD', 'ICBK', 'ICE',
'ICFI', 'ICHR', 'ICPT', 'ICUI', 'IDA', 'IDCC', 'IDN', 'IDT', 'IDXX', 'IDYA',
'IESC', 'IEX', 'IFF', 'IGMS', 'IGT', 'IHC', 'IHRT', 'III', 'IIIN', 'IIIV',
'IIN', 'IIVI', 'ILMN', 'IMAX', 'IMGN', 'IMKTA', 'IMMR', 'IMMU', 'IMRA', 'IMUX',
'IMVT', 'IMXI', 'INBK', 'INCY', 'INDB', 'INFN', 'INFO', 'INFU', 'INGN', 'INGR',
'INO', 'INOV', 'INS', 'INSG', 'INSM', 'INSP', 'INSW', 'INT', 'INTC', 'INTU',
'INVA', 'IONS', 'IOSP', 'IOVA', 'IP', 'IPAR', 'IPG', 'IPGP', 'IPHI', 'IPI',
'IQV', 'IR', 'IRBT', 'IRDM', 'IRMD', 'IRTC', 'IRWD', 'ISBC', 'ISEE', 'ISRG',
'ISTR', 'IT', 'ITCI', 'ITGR', 'ITI', 'ITIC', 'ITRI', 'ITT', 'ITW', 'IVAC',
'IVC', 'IVZ', 'J', 'JACK', 'JAZZ', 'JBHT', 'JBL', 'JBLU', 'JBSS', 'JBT', 'JCI',
'JCOM', 'JEF', 'JELD', 'JJSF', 'JKHY', 'JLL', 'JNCE', 'JNJ', 'JNPR', 'JOE',
'JOUT', 'JPM', 'JRVR', 'JW/A', 'JWN', 'JYNT', 'K', 'KAI', 'KALA', 'KALU',
'KALV', 'KAMN', 'KAR', 'KBAL', 'KBH', 'KBR', 'KDMN', 'KDP', 'KE', 'KELYA',
'KERN', 'KEX', 'KEY', 'KEYS', 'KFRC', 'KFY', 'KHC', 'KIDS', 'KIN', 'KKR',
'KLAC', 'KLDO', 'KMB', 'KMI', 'KMPR', 'KMT', 'KMX', 'KN', 'KNL', 'KNSA',
'KNSL', 'KNX', 'KO', 'KOD', 'KODK', 'KOP', 'KOS', 'KPTI', 'KR', 'KRA', 'KRMD',
'KRNY', 'KRO', 'KROS', 'KRTX', 'KRUS', 'KRYS', 'KSS', 'KSU', 'KTB', 'KTOS',
'KURA', 'KVHI', 'KW', 'KWR', 'KZR', 'L', 'LAD', 'LAKE', 'LANC', 'LARK', 'LASR',
'LAUR', 'LAWS', 'LAZ', 'LB', 'LBAI', 'LBC', 'LBRDA', 'LBRDK', 'LBRT', 'LC',
'LCI', 'LCII', 'LCNB', 'LCUT', 'LDL', 'LDOS', 'LE', 'LEA', 'LECO', 'LEG',
'LEGH', 'LEN/B', 'LEN', 'LEVL', 'LFUS', 'LFVN', 'LGF/A', 'LGF/B', 'LGIH',
'LGND', 'LH', 'LHCG', 'LHX', 'LII', 'LILA', 'LILAK', 'LIN', 'LIND', 'LITE',
'LIVN', 'LIVX', 'LJPC', 'LKFN', 'LKQ', 'LL', 'LLNW', 'LLY', 'LMAT', 'LMNR',
'LMNX', 'LMST', 'LMT', 'LNC', 'LNDC', 'LNG', 'LNN', 'LNT', 'LNTH', 'LOB',
'LOCO', 'LOGC', 'LOGM', 'LOPE', 'LORL', 'LOVE', 'LOW', 'LPG', 'LPLA', 'LPSN',
'LPX', 'LQDA', 'LQDT', 'LRCX', 'LRN', 'LSCC', 'LSTR', 'LTHM', 'LTRPA', 'LULU',
'LUNA', 'LUV', 'LVGO', 'LVS', 'LW', 'LXFR', 'LXRX', 'LYB', 'LYFT', 'LYRA',
'LYTS', 'LYV', 'LZB', 'M', 'MA', 'MAN', 'MANH', 'MANT', 'MAR', 'MAS', 'MASI',
'MAT', 'MATW', 'MATX', 'MAXR', 'MBCN', 'MBI', 'MBII', 'MBIN', 'MBIO', 'MBUU',
'MBWM', 'MC', 'MCB', 'MCBC', 'MCBS', 'MCD', 'MCF', 'MCFT', 'MCHP', 'MCK',
'MCO', 'MCRB', 'MCRI', 'MCS', 'MCY', 'MD', 'MDB', 'MDC', 'MDGL', 'MDLA',
'MDLZ', 'MDP', 'MDRX', 'MDT', 'MDU', 'MEC', 'MED', 'MEDP', 'MEET', 'MEI',
'MEIP', 'MESA', 'MET', 'MFNC', 'MG', 'MGEE', 'MGI', 'MGLN', 'MGM', 'MGNI',
'MGNX', 'MGPI', 'MGRC', 'MGTA', 'MGTX', 'MGY', 'MHH', 'MHK', 'MHO', 'MIC',
'MIDD', 'MIK', 'MIME', 'MIRM', 'MITK', 'MJCO', 'MKC', 'MKL', 'MKSI', 'MKTX',
'MLAB', 'MLHR', 'MLI', 'MLM', 'MLP', 'MLR', 'MLSS', 'MMAC', 'MMC', 'MMI',
'MMM', 'MMS', 'MMSI', 'MNK', 'MNKD', 'MNLO', 'MNOV', 'MNRL', 'MNRO', 'MNSB',
'MNST', 'MNTA', 'MO', 'MOBL', 'MOD', 'MODN', 'MOFG', 'MOG/A', 'MOH', 'MORF',
'MORN', 'MOS', 'MOV', 'MPAA', 'MPB', 'MPC', 'MPWR', 'MPX', 'MR', 'MRBK', 'MRC',
'MRCY', 'MRK', 'MRKR', 'MRLN', 'MRNA', 'MRNS', 'MRO', 'MRSN', 'MRTN', 'MRTX',
'MRVL', 'MS', 'MSA', 'MSBI', 'MSCI', 'MSEX', 'MSFT', 'MSGE', 'MSGN', 'MSGS',
'MSI', 'MSM', 'MSON', 'MSTR', 'MTB', 'MTCH', 'MTD', 'MTDR', 'MTEM', 'MTG',
'MTH', 'MTN', 'MTOR', 'MTRN', 'MTRX', 'MTSC', 'MTSI', 'MTW', 'MTX', 'MTZ',
'MU', 'MUR', 'MUSA', 'MVBF', 'MWA', 'MXIM', 'MXL', 'MYE', 'MYFW', 'MYGN',
'MYL', 'MYOK', 'MYRG', 'NAT', 'NATH', 'NATI', 'NATR', 'NAV', 'NAVI', 'NBEV',
'NBHC', 'NBIX', 'NBL', 'NBN', 'NBR', 'NBSE', 'NBTB', 'NC', 'NCBS', 'NCLH',
'NCMI', 'NCR', 'NDAQ', 'NDLS', 'NDSN', 'NEE', 'NEM', 'NEO', 'NEOG', 'NERV',
'NESR', 'NET', 'NEU', 'NEWR', 'NEX', 'NEXT', 'NFBK', 'NFG', 'NFLX', 'NG',
'NGHC', 'NGM', 'NGVC', 'NGVT', 'NH', 'NHC', 'NI', 'NJR', 'NK', 'NKE', 'NKSH',
'NKTR', 'NL', 'NLOK', 'NLS', 'NLSN', 'NLTX', 'NMIH', 'NMRD', 'NMRK', 'NNBR',
'NNI', 'NOC', 'NODK', 'NOV', 'NOVA', 'NOVT', 'NOW', 'NP', 'NPK', 'NPO', 'NPTN',
'NR', 'NRBO', 'NRC', 'NRG', 'NRIM', 'NSC', 'NSCO', 'NSIT', 'NSP', 'NSSC',
'NSTG', 'NTAP', 'NTB', 'NTCT', 'NTGR', 'NTLA', 'NTNX', 'NTRA', 'NTRS', 'NTUS',
'NUAN', 'NUE', 'NUS', 'NUVA', 'NVAX', 'NVCR', 'NVDA', 'NVEC', 'NVEE', 'NVR',
'NVRO', 'NVST', 'NVT', 'NVTA', 'NWBI', 'NWE', 'NWFL', 'NWL', 'NWLI', 'NWN',
'NWPX', 'NWS', 'NWSA', 'NX', 'NXGN', 'NXST', 'NXTC', 'NYCB', 'NYMX', 'NYT',
'OBNK', 'OC', 'OCFC', 'OCUL', 'OCX', 'ODC', 'ODFL', 'ODP', 'ODT', 'OEC',
'OESX', 'OFED', 'OFG', 'OFIX', 'OFLX', 'OGE', 'OGS', 'OI', 'OII', 'OIS', 'OKE',
'OKTA', 'OLED', 'OLLI', 'OLN', 'OMC', 'OMCL', 'OMER', 'OMF', 'OMI', 'ON',
'ONB', 'ONEM', 'ONEW', 'ONTO', 'OOMA', 'OPBK', 'OPCH', 'OPK', 'OPRT', 'OPRX',
'OPTN', 'OPY', 'ORA', 'ORBC', 'ORCL', 'ORGO', 'ORGS', 'ORI', 'ORIC', 'ORLY',
'ORRF', 'OSBC', 'OSG', 'OSIS', 'OSK', 'OSMT', 'OSPN', 'OSTK', 'OSUR', 'OSW',
'OTIS', 'OTRK', 'OTTR', 'OVBC', 'OVID', 'OVLY', 'OVV', 'OXM', 'OXY', 'OYST',
'OZK', 'PACB', 'PACK', 'PACW', 'PAE', 'PAG', 'PAHC', 'PANL', 'PANW', 'PAR',
'PARR', 'PASG', 'PATK', 'PAVM', 'PAYC', 'PAYS', 'PAYX', 'PB', 'PBCT', 'PBF',
'PBFS', 'PBH', 'PBI', 'PBIP', 'PBYI', 'PCAR', 'PCB', 'PCG', 'PCRX', 'PCSB',
'PCTI', 'PCTY', 'PCYG', 'PCYO', 'PD', 'PDCE', 'PDCO', 'PDFS', 'PDLB', 'PDLI',
'PE', 'PEBK', 'PEBO', 'PEG', 'PEGA', 'PEN', 'PENN', 'PEP', 'PETQ', 'PETS',
'PFBC', 'PFBI', 'PFC', 'PFE', 'PFG', 'PFGC', 'PFHD', 'PFIS', 'PFNX', 'PFPT',
'PFS', 'PFSI', 'PFSW', 'PG', 'PGC', 'PGEN', 'PGNY', 'PGR', 'PGTI', 'PH',
'PHAS', 'PHAT', 'PHM', 'PHR', 'PI', 'PICO', 'PII', 'PINC', 'PING', 'PINS',
'PIPR', 'PIRS', 'PJT', 'PKBK', 'PKE', 'PKG', 'PKI', 'PKOH', 'PLAB', 'PLAN',
'PLAY', 'PLBC', 'PLCE', 'PLMR', 'PLNT', 'PLOW', 'PLPC', 'PLSE', 'PLT', 'PLUG',
'PLUS', 'PLXS', 'PM', 'PNC', 'PNFP', 'PNM', 'PNR', 'PNRG', 'PNTG', 'PNW',
'PODD', 'POOL', 'POR', 'POST', 'POWI', 'POWL', 'PPBI', 'PPC', 'PPD', 'PPG',
'PPL', 'PQG', 'PRA', 'PRAA', 'PRAH', 'PRDO', 'PRFT', 'PRGO', 'PRGS', 'PRI',
'PRIM', 'PRK', 'PRLB', 'PRMW', 'PRNB', 'PRO', 'PROS', 'PROV', 'PRPL', 'PRSC',
'PRSP', 'PRTA', 'PRTH', 'PRTK', 'PRTS', 'PRU', 'PRVB', 'PRVL', 'PS', 'PSMT',
'PSN', 'PSNL', 'PSTG', 'PSX', 'PTC', 'PTCT', 'PTEN', 'PTGX', 'PTON', 'PTRS',
'PTSI', 'PTVCB', 'PUMP', 'PVAC', 'PVBC', 'PVH', 'PWFL', 'PWOD', 'PWR', 'PXD',
'PXLW', 'PYPL', 'PZN', 'PZZA', 'QADA', 'QCOM', 'QCRH', 'QDEL', 'QGEN', 'QLYS',
'QMCO', 'QNST', 'QRTEA', 'QRVO', 'QTNT', 'QTRX', 'QTWO', 'QUAD', 'QUOT', 'R',
'RAD', 'RAMP', 'RAPT', 'RARE', 'RAVN', 'RBB', 'RBBN', 'RBC', 'RBCAA', 'RBNC',
'RCII', 'RCKT', 'RCKY', 'RCL', 'RCM', 'RCUS', 'RDFN', 'RDN', 'RDNT', 'RDUS',
'RDVT', 'RE', 'REAL', 'REFR', 'REGI', 'REGN', 'REPH', 'REPL', 'RES', 'RESN',
'RETA', 'REV', 'REVG', 'REX', 'REYN', 'REZI', 'RF', 'RFL', 'RGA', 'RGCO',
'RGEN', 'RGLD', 'RGNX', 'RGP', 'RGR', 'RGS', 'RH', 'RHI', 'RICK', 'RIG',
'RIGL', 'RILY', 'RJF', 'RL', 'RLGT', 'RLGY', 'RLI', 'RLMD', 'RM', 'RMAX',
'RMBI', 'RMBS', 'RMD', 'RMNI', 'RMR', 'RMTI', 'RNG', 'RNR', 'RNST', 'ROAD',
'ROCK', 'ROG', 'ROK', 'ROKU', 'ROL', 'ROLL', 'ROP', 'ROST', 'RP', 'RPAY',
'RPD', 'RPM', 'RRBI', 'RRC', 'RRGB', 'RRR', 'RS', 'RSG', 'RST', 'RTRX', 'RTX',
'RUBY', 'RUN', 'RUSHA', 'RUSHB', 'RUTH', 'RVMD', 'RVNC', 'RVP', 'RVSB', 'RXN',
'RYAM', 'RYI', 'RYTM', 'SABR', 'SAFM', 'SAFT', 'SAGE', 'SAH', 'SAIA', 'SAIC',
'SAIL', 'SAL', 'SALT', 'SAM', 'SAMG', 'SANM', 'SASR', 'SATS', 'SAVA', 'SAVE',
'SB', 'SBBP', 'SBCF', 'SBFG', 'SBGI', 'SBH', 'SBNY', 'SBSI', 'SBT', 'SBUX',
'SC', 'SCCO', 'SCHL', 'SCHN', 'SCHW', 'SCI', 'SCL', 'SCOR', 'SCPH', 'SCS',
'SCSC', 'SCU', 'SCVL', 'SCWX', 'SDGR', 'SEAC', 'SEAS', 'SEB', 'SEDG', 'SEE',
'SEIC', 'SELB', 'SEM', 'SENEA', 'SERV', 'SF', 'SFBS', 'SFE', 'SFIX', 'SFL',
'SFM', 'SFNC', 'SFST', 'SGA', 'SGC', 'SGEN', 'SGH', 'SGMO', 'SGMS', 'SGRY',
'SHAK', 'SHBI', 'SHEN', 'SHOO', 'SHW', 'SHYF', 'SI', 'SIBN', 'SIEB', 'SIEN',
'SIG', 'SIGA', 'SIGI', 'SILK', 'SIRI', 'SITE', 'SITM', 'SIVB', 'SIX', 'SJI',
'SJM', 'SJW', 'SKX', 'SKY', 'SKYW', 'SLAB', 'SLB', 'SLCA', 'SLCT', 'SLDB',
'SLGN', 'SLM', 'SLNO', 'SLP', 'SM', 'SMAR', 'SMBC', 'SMBK', 'SMCI', 'SMED',
'SMG', 'SMMF', 'SMP', 'SMPL', 'SMSI', 'SMTC', 'SNA', 'SNBR', 'SNCR', 'SNDR',
'SNDX', 'SNEX', 'SNFCA', 'SNPS', 'SNV', 'SNX', 'SO', 'SOI', 'SOLY', 'SON',
'SONA', 'SONO', 'SP', 'SPB', 'SPCE', 'SPFI', 'SPGI', 'SPKE', 'SPLK', 'SPNE',
'SPNS', 'SPOK', 'SPOT', 'SPPI', 'SPR', 'SPRO', 'SPSC', 'SPT', 'SPTN', 'SPWH',
'SPWR', 'SPXC', 'SQ', 'SR', 'SRCE', 'SRCL', 'SRDX', 'SRE', 'SREV', 'SRGA',
'SRI', 'SRNE', 'SRPT', 'SRRK', 'SRT', 'SSB', 'SSD', 'SSNC', 'SSP', 'SSTI',
'SSTK', 'ST', 'STAA', 'STBA', 'STC', 'STE', 'STFC', 'STL', 'STLD', 'STMP',
'STND', 'STNE', 'STNG', 'STOK', 'STRA', 'STRL', 'STRO', 'STRS', 'STSA', 'STT',
'STXB', 'STXS', 'STZ', 'SUM', 'SUPN', 'SVMK', 'SVRA', 'SWAV', 'SWBI', 'SWCH',
'SWI', 'SWK', 'SWKH', 'SWKS', 'SWM', 'SWN', 'SWTX', 'SWX', 'SXC', 'SXI', 'SXT',
'SYBT', 'SYF', 'SYK', 'SYKE', 'SYNA', 'SYNH', 'SYRS', 'SYX', 'SYY', 'T',
'TACO', 'TALO', 'TAP', 'TARA', 'TAST', 'TBBK', 'TBI', 'TBIO', 'TBK', 'TBNK',
'TBPH', 'TCBI', 'TCBK', 'TCDA', 'TCF', 'TCFC', 'TCI', 'TCMD', 'TCRR', 'TCS',
'TCX', 'TDC', 'TDG', 'TDOC', 'TDS', 'TDW', 'TDY', 'TEAM', 'TECH', 'TELA',
'TELL', 'TEN', 'TENB', 'TER', 'TEX', 'TFC', 'TFSL', 'TFX', 'TG', 'TGH', 'TGI',
'TGNA', 'TGT', 'TGTX', 'TH', 'THC', 'THFF', 'THG', 'THO', 'THR', 'THRM', 'THS',
'TIF', 'TILE', 'TIPT', 'TISI', 'TITN', 'TJX', 'TKR', 'TLYS', 'TMDX', 'TMHC',
'TMO', 'TMP', 'TMST', 'TMUS', 'TNAV', 'TNC', 'TNDM', 'TNET', 'TOL', 'TOWN',
'TPB', 'TPC', 'TPCO', 'TPH', 'TPIC', 'TPR', 'TPRE', 'TPTX', 'TPX', 'TR', 'TRC',
'TREC', 'TREE', 'TREX', 'TRGP', 'TRHC', 'TRIP', 'TRMB', 'TRMK', 'TRN', 'TRNS',
'TROW', 'TROX', 'TRS', 'TRST', 'TRTN', 'TRU', 'TRUE', 'TRUP', 'TRV', 'TRWH',
'TSBK', 'TSC', 'TSCO', 'TSE', 'TSLA', 'TSN', 'TT', 'TTC', 'TTD', 'TTEC',
'TTEK', 'TTGT', 'TTMI', 'TTWO', 'TUP', 'TVTY', 'TW', 'TWLO', 'TWNK', 'TWOU',
'TWST', 'TWTR', 'TXG', 'TXMD', 'TXN', 'TXRH', 'TXT', 'TYL', 'TYME', 'UA',
'UAA', 'UAL', 'UBER', 'UBFO', 'UBSI', 'UBX', 'UCBI', 'UCTT', 'UEC', 'UEIC',
'UFCS', 'UFI', 'UFPI', 'UFPT', 'UFS', 'UGI', 'UHAL', 'UHS', 'UI', 'UIHC',
'UIS', 'ULBI', 'ULH', 'ULTA', 'UMBF', 'UMPQ', 'UNF', 'UNFI', 'UNH', 'UNM',
'UNP', 'UNTY', 'UNVR', 'UPLD', 'UPS', 'UPWK', 'URBN', 'URGN', 'URI', 'USB',
'USCR', 'USFD', 'USLM', 'USM', 'USNA', 'USPH', 'USX', 'UTHR', 'UTI', 'UTL',
'UTMD', 'UUUU', 'UVE', 'UVSP', 'UVV', 'V', 'VAC', 'VALU', 'VAPO', 'VAR',
'VBIV', 'VBTX', 'VC', 'VCEL', 'VCRA', 'VCYT', 'VEC', 'VECO', 'VEEV', 'VERI',
'VERO', 'VERU', 'VERY', 'VFC', 'VG', 'VGR', 'VHC', 'VIAC', 'VIACA', 'VIAV',
'VICR', 'VIE', 'VIR', 'VIRT', 'VIVO', 'VKTX', 'VLGEA', 'VLO', 'VLY', 'VMC',
'VMD', 'VMI', 'VMW', 'VNDA', 'VNRX', 'VOXX', 'VOYA', 'VPG', 'VRA', 'VRAY',
'VRCA', 'VREX', 'VRNS', 'VRNT', 'VRRM', 'VRS', 'VRSK', 'VRSN', 'VRT', 'VRTS',
'VRTU', 'VRTV', 'VRTX', 'VSAT', 'VSEC', 'VSH', 'VSLR', 'VST', 'VSTM', 'VSTO',
'VTOL', 'VTVT', 'VVI', 'VVNT', 'VVV', 'VXRT', 'VYGR', 'VZ', 'W', 'WAB', 'WABC',
'WAFD', 'WAL', 'WASH', 'WAT', 'WBA', 'WBS', 'WBT', 'WCC', 'WD', 'WDAY', 'WDC',
'WDFC', 'WDR', 'WEC', 'WEN', 'WERN', 'WETF', 'WEX', 'WEYS', 'WFC', 'WGO', 'WH',
'WHD', 'WHG', 'WHR', 'WIFI', 'WINA', 'WING', 'WIRE', 'WK', 'WKHS', 'WLDN',
'WLFC', 'WLK', 'WLL', 'WLTW', 'WM', 'WMB', 'WMGI', 'WMK', 'WMS', 'WMT', 'WNC',
'WNEB', 'WOR', 'WORK', 'WOW', 'WPX', 'WRB', 'WRK', 'WRLD', 'WRTC', 'WSBC',
'WSBF', 'WSC', 'WSFS', 'WSM', 'WSO', 'WST', 'WTBA', 'WTFC', 'WTI', 'WTM',
'WTRE', 'WTRG', 'WTRH', 'WTS', 'WTTR', 'WU', 'WVE', 'WW', 'WWD', 'WWE', 'WWW',
'WYND', 'WYNN', 'X', 'XAIR', 'XBIT', 'XCUR', 'XEC', 'XEL', 'XENT', 'XERS',
'XFOR', 'XGN', 'XLNX', 'XLRN', 'XNCR', 'XOM', 'XOMA', 'XONE', 'XPEL', 'XPER',
'XPO', 'XRAY', 'XRX', 'XYL', 'Y', 'YELP', 'YETI', 'YEXT', 'YMAB', 'YORW',
'YUM', 'YUMC', 'Z', 'ZBH', 'ZBRA', 'ZEN', 'ZEUS', 'ZG', 'ZGNX', 'ZION', 'ZIOP',
'ZIXI', 'ZM', 'ZNGA', 'ZNTL', 'ZS', 'ZTS', 'ZUMZ', 'ZUO', 'ZYXI']
"""
