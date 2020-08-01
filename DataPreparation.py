# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 2020
Author: Brandi Beals
Description: Thesis Data Preparation
"""

######################################
# IMPORT PACKAGES
######################################

import os
import pypyodbc
import pandas as pd

######################################
# DEFIINITIONS
######################################

os.chdir(r'C:\Users\bbeals\Dropbox (Personal)\Masters in Predictive Analytics\590-Thesis\NW590-Thesis')

sql = """SELECT TOP (1000) *
  FROM edm.ConsensusEstimatesAnnualFactset f
  JOIN edm.ConstituentList ON ConstituentList.edmSecurityId = f.edmSecurityId
  JOIN edm.InvestmentTeam ON InvestmentTeam.id = ConstituentList.investmentTeamId
  JOIN edm.SecurityReference ON SecurityReference.edmSecurityId = ConstituentList.edmSecurityId
  JOIN edm.SecurityType ON SecurityType.id = SecurityReference.securityTypeId
  WHERE InvestmentTeamCode='GRW'
  AND effectiveDate>'3-31-2020'
  AND SecurityType.code='COM'
  AND f.edmSecurityId='70001937'
  ORDER BY effectiveDate, f.edmSecurityId DESC"""

######################################
# GET DATA
######################################

def getdata(sql):
    connection = pypyodbc.connect("Driver={SQL Server Native Client 11.0};"
                            "Server=milsql02;"
                            "Database=ArtisanApplication;"
                            "Trusted_Connection=yes;")
    df = pd.read_sql_query(sql, connection)
    df.head()

def main():
    getdata(sql)

if __name__ == "__main__":
    main()