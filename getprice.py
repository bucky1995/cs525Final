import boto3
import json

class LambdaPrice:
    def get_price(self):
        client=boto3.client('pricing')
        response = client.get_products(
            ServiceCode='AWSLambda',
        )

        priceList=response["PriceList"]

        list1=[]


        for i in range(len(priceList)):
            lambdaProduct=json.loads(priceList[i])['terms']['OnDemand']
            for i in lambdaProduct:
                lambdaProduct2=lambdaProduct[i]['priceDimensions']
                for i in lambdaProduct2:
                    list1.append(float(lambdaProduct2[i]['pricePerUnit']['USD']))


        list2=[]




        for i in range(len(priceList)):
            lambdaProduct=json.loads(priceList[i])['product']['attributes']['group']
            list2.append(lambdaProduct)



        dict2={}

        for i in range(len(list1)):
            if list1[i]>0:
                if list2[i] in dict2:
                    if list1[i] < dict2[list2[i]]:
                        dict2[list2[i]] = list1[i]
                else:
                    dict2[list2[i]] = list1[i]
        return dict2


