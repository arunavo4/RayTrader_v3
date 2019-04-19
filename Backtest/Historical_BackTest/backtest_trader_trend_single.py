"""
@########################### Same Trend Strategy But With Single Stock ###############################

This strategy is based on the fact that, when there is a sudden change in trend it will  continue
more than one day.

So we need to look for a falling trend or a buying trend continuous for at least 3-4 days
and then there is a change in trend. wait for one day to confirm
now check these in a day candle. and select the stocks

On the day of trading , open a 5 min candle .
In the first 5 min candle we need to check if its green , we buy 5-10 paisa over the High of first candle
and the mid of the candle is the stop-loss and the range of the first candle is the target.

"""


