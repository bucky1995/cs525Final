from getprice import LambdaPrice

gp=LambdaPrice()

priceList=gp.get_price()

def normal_pricing(list):
    Total_GBs=0
    Total_request=0
    for l in list:
        Total_GBs+=l[0]*l[1]*l[2]/1024
        Total_request+=l[0]
    calculate_price=(Total_GBs-400000)*priceList['AWS-Lambda-Duration']
    request_price=(Total_request-1000000)*priceList['AWS-Lambda-Requests']
    return calculate_price+request_price

#request numbers, time(s), memory(MB)

list=[[25000000,0.2,128],[5000000,0.5,448],[2500000,1,1024]]

print(normal_pricing(list))

def concurrent_pricing(list):
    Concurrent_GBs=0
    Calculate_GBs=0
    Request=0
    for l in list:
        Concurrent_GBs+=l[0]*l[1]*l[2]/1024
        Calculate_GBs+=l[2]*l[3]*l[4]/1024
        Request+=l[3]
    concurrent_price=Concurrent_GBs*priceList['AWS-Lambda-Provisioned-Concurrency']
    request_price=Request*priceList['AWS-Lambda-Requests']
    calculate_price=Calculate_GBs*priceList['AWS-Lambda-Duration-Provisioned']
    return concurrent_price+request_price+calculate_price

#concurrent number, time(s), memory(MB), request number, request time(s)

list2=[[1000,7200,1024,1200000,1]]
print(concurrent_pricing(list2))