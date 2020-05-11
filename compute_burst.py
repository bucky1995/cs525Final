def burst_week_request(burst_day, burst_hour,idel_hour, instance_baseline, instance_burst,instance_price):
    request = burst_day * (idel_hour * 60 * 60 * instance_baseline + burst_hour * 60 * 60 * instance_burst) + (
                7 - burst_day) * 24 * 60 * 60 * instance_baseline
    price = 7*24*instance_price
    return request,price
def burst_day_request(burst_hour,idel_hour, instance_baseline, instance_burst,instance_price):
    request = idel_hour * 60 * 60 * instance_baseline + burst_hour * 60 * 60 * instance_burst
    price = 24*instance_price
    return request,price
def burst_month_reqeust(burst_day, burst_hour,idel_hour, instance_baseline, instance_burst,instance_price):
    request = burst_day * (idel_hour * 60 * 60 * instance_baseline + burst_hour * 60 * 60 * instance_burst) + (
                30 - burst_day) * 24 * 60 * 60 * instance_baseline
    price = 30*24*instance_price
    return request,price

idel_hour = 14
burst_hour = 10
burst_week_day = 2
burst_month_day = 10

t3 = (("nano",0.0052, 1723, 86, 0.5), ("micro", 0.0104,1823,176,1.0), ("small",0.0208,1746, 278, 2.0), ("medium", 0.0416,1735,325,4.0), ("large", 0.1664, 1832, 587, 8.0))
for instance_type, instance_price, instance_burst, instance_baseline, instance_ram in t3:
    print(instance_type)
    day_request, day_price = burst_day_request(burst_hour,idel_hour,instance_baseline,instance_burst, instance_price)
    week_request, week_price = burst_week_request(burst_week_day,burst_hour, idel_hour,instance_baseline, instance_burst, instance_price)
    month_request, month_price = burst_month_reqeust(burst_month_day,burst_hour,idel_hour, instance_baseline, instance_burst, instance_price)
    print(day_request ,round(day_price,2))
    print(week_request, round(week_price,2))
    print(month_request,  round(month_price,2))



