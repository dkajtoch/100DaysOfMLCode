McKinsey Analytics Online Hackaton (20 - 22.07.2018)
==================================


# Problem Statement
Your client is an Insurance company and they need your help in building a model to predict the propensity to pay renewal premium and build an incentive plan for its agents to maximise the net revenue (i.e. renewals - incentives given to collect the renewals) collected from the policies post their issuance.
 
You have information about past transactions from the policy holders along with their demographics. The client has provided aggregated historical transactional data like number of premiums delayed by 3/6/12 months across all the products, number of premiums paid, customer sourcing channel and customer demographics like age, monthly income and area type.
 
In addition to the information above, the client has provided the following relationships:
1. Expected effort in hours put in by an agent for incentives provided.
2. Expected increase in chances of renewal, given the effort from the agent.
 
Given the information, the client wants you to predict the propensity of renewal collection and create an incentive plan for agents (at policy level) to maximise the net revenues from these policies.

# My Solution
<img src="./figures/model_scheme.svg" width="800" height="400" />

Stacking using optimized XGBoost, Random Forest and Neural Network Classifiers. The 3 models were performing the best in terms of AUC and LogLoss.
Output probabilities were used as an input to the Logistic Regression classifier with and without extra feature. 

Public score:
* 0.729573704224991 (with extra feature)
* 0.729171265797949 (without extra feature)

### Missing values
There was no strategy in case of XGBoost. For other models I imputed missing values using mean of the columns. 

### Other
In case of Neural Network `Income` was transformed using `log(x)` function and `age_in_days` using Box-Cox transform (optimal). I also used `StandardScaler` for all features.

# Summary

<img src="./figures/score_summary.svg" width=800 height="400" />

# Results
* 2nd on the public leaderboard.
* 4th on the private leaderboard.
