#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import urllib.request
import os

try:
	import hjson
except:
	print('Install hjson with "pip3 install hjson"')
	quit()
try:
	from tabulate import tabulate
except:
	print('Install tabulate with "pip3 install tabulate"')
	quit()

def get_prices(instanceType, OS, AMI, performance):
	
	req = urllib.request.Request(
		AMI, 
		data=None,
		headers={
			'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:73.0) Gecko/20100101 Firefox/73.0'
		}
	)
	
	page = urllib.request.urlopen(req)
	page_content = page.read().decode('utf-8')
	cd = page_content[page_content.find('/*<![CDATA[*/') + 13:page_content.find('/*]]>*/',page_content.find('/*<![CDATA[*/') + 13)]
	productId = cd[cd.find('productId: "') + 12:cd.find('"',cd.find('productId: "') + 12)]
	hjson_string = cd[cd.find('var awsmpInitialState = ') + 24:cd.find('};',cd.find('var awsmpInitialState = ') + 24) + 1]
	hjson_dict = hjson.loads(hjson_string)
	
	for region in hjson_dict['offerIon']['offer']['pricing']['regions']:
		for idx, row in enumerate(hjson_dict['offerIon']['offer']['pricing']['regions'][region][productId]['hourly']['displayElements']['properties']['rows']):
			if hjson_dict['offerIon']['offer']['pricing']['regions'][region][productId]['hourly']['displayElements']['properties']['rows'][idx]['instanceType'] == instanceType:
				hourly_rate = float(hjson_dict['offerIon']['offer']['pricing']['regions'][region][productId]['hourly']['displayElements']['properties']['rows'][idx]['totalRate'])
				instanceRegion = instanceType + region
				try:
					idx = [instanceRegion for instanceRegion, InstanceType, Region, WindowsHourly, LinuxHourly, Windows100M, Linux100M in results].index(instanceRegion)
					if OS == 'Windows':
						results[idx][3] = hourly_rate
						results[idx][5] = round((100000000/performance)*(hourly_rate/3600), 3)
					else:
						results[idx][4] = hourly_rate
						results[idx][6] = round((100000000/performance)*(hourly_rate/3600), 3)
				except:
					if OS == 'Windows':
						results.append([instanceRegion, instanceType, region, hourly_rate, '', round((100000000/performance)*(hourly_rate/3600), 3), ''])
					else:
						results.append([instanceRegion, instanceType, region, '', hourly_rate, '', round((100000000/performance)*(hourly_rate/3600), 3)])

results_headers = ['Instance Type', 'Region', 'Windows $/hr', 'Linux $/hr', 'Windows $/100M passwords', 'Linux $/100M passwords']
results = []

instances = [
#Instance Type, OS, AMI link, Performance (passwords per second when cracking a Word 2019 document)
['g3s.xlarge', 'Windows', 'https://aws.amazon.com/marketplace/pp/B073WHLGMC', 4616],
['g3s.xlarge', 'Linux', 'https://aws.amazon.com/marketplace/pp/B07CQ33QKV', 4616],
['g4dn.xlarge', 'Windows', 'https://aws.amazon.com/marketplace/pp/B07TS3S3ZH', 8293],
['g4dn.xlarge', 'Linux', 'https://aws.amazon.com/marketplace/pp/B07YV3B14W', 8293],
['p3.2xlarge', 'Linux', 'https://aws.amazon.com/marketplace/pp/B07CQ33QKV', 21088],
['p3.2xlarge', 'Windows', 'https://aws.amazon.com/marketplace/pp/B07TS3S3ZH', 21088]
]

for instance in instances:
	get_prices(instance[0], instance[1], instance[2], instance[3], )

results = [i[1:] for i in results]
results.sort()

print(tabulate(results, headers = results_headers, stralign = 'decimal'))

ap_cheapest_os = ''
eu_cheapest_os = ''
us_cheapest_os = ''
ap_cheapest_index = 0
eu_cheapest_index = 0
us_cheapest_index = 0
ap_cheapest_100Mprice = 999
eu_cheapest_100Mprice = 999
us_cheapest_100Mprice = 999
ap_cheapest_hourlyprice = 0
eu_cheapest_hourlyprice = 0
us_cheapest_hourlyprice = 0


for i, x in enumerate(results):
	if (x[1][:2] == 'ap') and (x[4] != '') and (x[4] < ap_cheapest_100Mprice):
		ap_cheapest_os = 'Windows'
		ap_cheapest_index = i
		ap_cheapest_100Mprice = x[4]
		ap_cheapest_hourlyprice = x[2]
	if (x[1][:2] == 'ap') and (x[5] != '') and (x[5] < ap_cheapest_100Mprice):
		ap_cheapest_os = 'Linux'
		ap_cheapest_index = i
		ap_cheapest_100Mprice = x[5]
		ap_cheapest_hourlyprice = x[3]
	if (x[1][:2] == 'eu') and (x[4] != '') and (x[4] < eu_cheapest_100Mprice):
		eu_cheapest_os = 'Windows'
		eu_cheapest_index = i
		eu_cheapest_100Mprice = x[4]
		eu_cheapest_hourlyprice = x[2]
	if (x[1][:2] == 'eu') and (x[5] != '') and (x[5] < eu_cheapest_100Mprice):
		eu_cheapest_os = 'Linux'
		eu_cheapest_index = i
		eu_cheapest_100Mprice = x[5]
		eu_cheapest_hourlyprice = x[3]
	if (x[1][:2] == 'us') and (x[4] != '') and (x[4] < us_cheapest_100Mprice):
		us_cheapest_os = 'Windows'
		us_cheapest_index = i
		us_cheapest_100Mprice = x[4]
		us_cheapest_hourlyprice = x[2]
	if (x[1][:2] == 'us') and (x[5] != '') and (x[5] < us_cheapest_100Mprice):
		us_cheapest_os = 'Linux'
		us_cheapest_index = i
		us_cheapest_100Mprice = x[5]
		us_cheapest_hourlyprice = x[3]

g2 = 4703
p2 = 2019
g4 = 8304
p3 = 21076

print()
val = input("Target passwords number /M: ")
print(val)
if(float(val) < 7.28):
	print("The most ecnomic choice is " + str(results[ap_cheapest_index][0])+ ", and the total cost will around $ " +str(us_cheapest_hourlyprice))
	print("The most efficient choice is " + str(results[ap_cheapest_index][0])+ ", and the total cost will around $ " +str(us_cheapest_hourlyprice))
if(float(val) >= 7.28):
	time_g4 = float(val)*1000000/(g4*3600)
	price_g4 = time_g4 * us_cheapest_hourlyprice
	time_p3 = float(val)*1000000/(p3*3600)
	price_p3 = time_p3 * x[4]
	print("1. The most economic choice is: g4dn.xlarge" + " total cost would be: "+str(price_g4) + " time: " + str(time_g4) +" hour")
	print("2. The best performance choice is: p3.2xlarge" + " total cost would be: "+str(price_p3) + " time: " + str(time_p3) +" hour")
choice = input("Select which instance to run, enter the number of choice")
if(choice == "1"):
	instance_name = input("Name a instance: ")
	print(" ")
	command = "docker-machine create --driver amazonec2 \
                      --amazonec2-region eu-west-1 \
                      --amazonec2-ami ami-58d7e821 \
                      --amazonec2-instance-type g4dn.xlarge \
                      --amazonec2-vpc-id vpc-abc\ "
	command = command + instance_name
	print(os.system(command))
if(choice == "2"):
	instance_name = input("Name a instance: ")
	print(" ")
	command = "docker-machine create --driver amazonec2 \
                      --amazonec2-region eu-west-1 \
                      --amazonec2-ami ami-58d7e821 \
                      --amazonec2-instance-type p3.xlarge \
                      --amazonec2-vpc-id vpc-abc\ "
	command = command + instance_name
	print(os.system(command))