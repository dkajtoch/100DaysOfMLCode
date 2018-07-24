McKinsey Analytics Online Hackaton (20 - 22.07.2018)
==================================


# Problem Statement
Your client is an Insurance company and they need your help in building a model to predict the propensity to pay renewal premium and build an incentive plan for its agents to maximise the net revenue (i.e. renewals - incentives given to collect the renewals) collected from the policies post their issuance.
 
You have information about past transactions from the policy holders along with their demographics. The client has provided aggregated historical transactional data like number of premiums delayed by 3/6/12 months across all the products, number of premiums paid, customer sourcing channel and customer demographics like age, monthly income and area type.
 
In addition to the information above, the client has provided the following relationships:
1. Expected effort in hours put in by an agent for incentives provided.
2. Expected increase in chances of renewal, given the effort from the agent.
 
Given the information, the client wants you to predict the propensity of renewal collection and create an incentive plan for agents (at policy level) to maximise the net revenues from these policies.

# Evaluation Criteria

Your solutions will be evaluated on 2 criteria:
* A. The base probability of receiving a premium on a policy without considering any incentive.
* B. The monthly incentives you will provide on each policy to maximize the net revenue. 

Part A:
The probabilities predicted by the participants would be evaluated using AUC ROC score.
 
Part B:
The net revenue across all policies will be calculated in the following manner:

$$\text{Total Net Revenue} = \sum\limits_{\text{sum across all policies}} \left[(p_{\text{benchmark}} + \Delta p) \cdot \text{premium on policy} - \text{Incentive on policy}\right],$$
where
* $$p_{\rm benchmark}$$ - is the renewal probability predicted using a benchmark model by the insurance company.
* $$\Delta p$$ - (% Improvement in renewal probability*$$p_{\rm benchmark}$$ ) is the improvement in renewal probability calculated from the agent
efforts in hours.
* `Premium on policy` is the premium paid by the policy holder for the policy in consideration.
* `Incentive on policy` is the incentive given to the agent for increasing the chance of renewal (estimated by the participant) for
each policy.

The following curve provide the relationship between extra effort in hours invested by the agent with Incentive to the agent and % improvement in renewal probability vs agent effort in hours.
 
1. Relationship b/w Extra efforts in hours invested by an agent and Incentive to agent. After a point more incentives does not convert to extra efforts.
    * $$y = 10\cdot(1 - \exp(-x/400))$$
2. Relationship between % improvement in renewal probability vs Agent effort in hours. The renewal probability cannot be improved beyond a certain level even with more efforts.
    * $$y = 20\cdot(1 - \exp(-x/5))$$

Overall Ranking at the leaderboard would be done using the following equation:
$$\text{Combined Score} = 0.7 \cdot \text{AUC-ROC} + 0.3 \cdot (\text{net revenue collected from all policies})\cdot\lambda,$$ 
where $$\lambda$$ is a normalizing factor.

Public leaderboard is based on 40% of the policies, while private leaderboard will be evaluated on remaining 60% of policies in the test dataset.


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

# Results
* 2nd on the public leaderboard.
* 4th on the private leaderboard.
