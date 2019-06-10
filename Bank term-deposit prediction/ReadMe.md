## Problem Statement
The PortugueseBank had run a telemarketing campaign in the past, making sales calls for a term-deposit product. Whether a prospect had bought the product or not is mentioned in the column named 'response'.
The marketing team wants to launch another campaign, and they want to learn from the past one. You, as an analyst, decide to build a supervised model in R/Python and achieve the following goals:

Reduce the marketing cost by X% and acquire Y% of the prospects (compared to random calling), where X and Y are to be maximized
Present the financial benefit of this project to the marketing team
Solution
'''
We can predict a prospect beforehand and thus only call only those customers who are likely to opt for a term deposit, 
Thus saving the cost of calling customers which would not have opted for a term deposit. 

Thus as can be seen from the output, we a total of around 7400(Positives)=4600(False positives)+2800(True positives) 

In random calling:    we get a total of 4640(True positives) out of 41,188 calls
In Selective calling: we get a total of 2800(True Positive) out of 7400 calls 

## Financial Benefit: 
Thus the model enables us to give 60.3% of the prospects in just ~17.9% cost and resources as compared to random calling.
'''