# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:05:50 2018

@author: rudas
"""

#Import the required libraries
from lxml import html
import requests
import random
import time
import pandas as pd
from fake_useragent import UserAgent

#Global declaration of user agent required for get() request
ua = UserAgent()

#Import the unique datasets present in our complaints database
filename = 'uniquezip_tx.csv'
data = pd.read_csv(filename, converters={0:str})

#Convert the dataframe into list
mnop = data['ZIP'].values.tolist()

#Change the subset values, in order to scrape data, like 1-500 records per day
mn = mnop[1435:1847]

#Create a nested list for gathering the scraped data for each zip
final = []
final_1 = []
for l in range(len(mn)):
final.append(final_1)

#for loop for multiple requests
for k in range(len(mn)):
agent = {'User-Agent': ua.random}
url = 'http://www.city-data.com/zips/'+mn[k]+'.html'
response = requests.get(url,headers=agent)

#Decode the website content using ISO 8859
decode_response = response.content.decode('ISO-8859-1')
tree = html.fromstring(decode_response)

#Obtain the required data from the html element using xpath
head_key = tree.xpath('//div[@id="body"][@class="row"]//following-sibling::b/following::text()'

#Create empty list to keep appending required values from the head_key
ab_list = []

for i in range(len(head_key)):
#Houses and condos
if("Houses and condos:" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])
#Housing units with mortgage
if("Housing units in zip code" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])

#Housing units without mortgage
if("Houses without a mortgage" in head_key[i]):
1
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])

#Estimated median house/condo value
if("Estimated median house/condo value in 2016" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])

#cost of living index
if("Mar. 2016 cost of living index in zip code" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])

#Estimated median household income in 2016:
if("Estimated median household income in 2016: " in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+2])

#Median gross rent
if("Median gross rent in 2016:" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])

#real estate property taxes paid for housing units with mortgages
if("Median real estate property taxes paid for housing units with mortgages" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])

#real estate property taxes paid for housing units with no mortgage
if("Median real estate property taxes paid for housing units with no mortgage" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])

#Unemployment rate
if("Unemployed:" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])

#Males
if("Males:" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])

#Females
if("Females:" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])

#resident age
if("Median resident age:" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+2])

#commute
if("Mean travel time to work (commute):" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])
2

#Population density
if("Population density:" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])

#Population density
if("% of renters here:" in head_key[i]):
#ab_list.append(head_key[i])
ab_list.append(head_key[i+1])
x = ab_list.copy()

#Cleansing the obtained values
for i in range(len(x)):
x[i] = x[i].replace('$','')
x[i] = x[i].replace('minutes','')
x[i] = x[i].replace('years','')
x[i] = x[i].replace('\r\n','')
x[i] = x[i].replace(' ','')
x[i] = x[i].replace('\xa0','')
x[i] = x[i].replace(',','')
x[i] = x[i].replace('%','')

#Further cleasning activity
y = x.copy()
for i in range(len(x)):
if("(" in x[i]):
y[i] = "(".join(x[i].split("(")[:-1])
final[k]=y.copy()

#Generate a random time delay between each web request
timeDelay = random.randrange(0, 25)
time.sleep(timeDelay)

#Create a new dataframe and export to csv file format
new_df = pd.DataFrame(columns=['houses','percent_rent', 'cost_of_living','pop_density', 'm_prop_tax_'male', 'female', 'unemp', 'commute', 'est_m_house_val', 'm_res_age', 'est_m_house_'houses_w_mtg', 'houses_w_out_mtg', 'm_gross_rent'], data=final, index=mn)


new_df.to_csv("ZipTX/zip_tx_v2.csv")
