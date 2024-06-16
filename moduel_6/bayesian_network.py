"""
BU MET CS 767 Assignment 6: Bayesian Network
Alessandro Allegranzi
6/19/2024
"""

# run pip install pgmpy in terminal to install pgmpy library.
# https://pgmpy.org/models/bayesiannetwork.html
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian network
model = BayesianNetwork([
    ('Economic Outlook', 'Political Affiliation'), 
    ('Political Affiliation', 'Vote'), 
    ('Economic Outlook', 'Vote')
])

# Defining the conditional probability distribution (CPD) for each node
cpd_economic_outlook = TabularCPD(variable='Economic Outlook', variable_card=2, values=[[0.6], [0.4]])


cpd_political_affiliation = TabularCPD(variable='Political Affiliation', variable_card=3, 
                                       values=[
                                           [0.6, 0.2],  # Higher probability of being conservative if economic outlook is negative
                                           [0.25, 0.5],  # Higher probability of being liberal if economic outlook is positive
                                           [0.15, 0.3]   # Independent remains less likely in both cases
                                       ],
                                       evidence=['Economic Outlook'],
                                       evidence_card=[2])

# Assuming 0: conservative, 1: liberal for vote
cpd_vote = TabularCPD(variable='Vote', variable_card=2, 
                      values=[
                          # Adjusted probabilities for Conservative (0) and Liberal (1) votes
                          # Order: [Conservative, Liberal, Independent] x [Negative, Positive] Economic Outlook
                          [0.4, 0.2, 0.5, 0.3, 0.6, 0.4],  # Conservative Vote
                          [0.6, 0.8, 0.5, 0.7, 0.4, 0.6]   # Liberal Vote
                      ],
                      evidence=['Political Affiliation', 'Economic Outlook'],
                      evidence_card=[3, 2])

# Add CPDs to the model
model.add_cpds(cpd_economic_outlook, cpd_political_affiliation, cpd_vote)

# Check if the model is valid
assert model.check_model()

# Perform inference
inference = VariableElimination(model)

# Query the model: P(Vote | Economic Outlook=1, Political Affiliation=1)
result = inference.query(variables=['Political Affiliation'], evidence={'Economic Outlook': 1, 'Vote': 1})
# Second example is negative economic outlook and liberal political affiliation.
result_two = inference.query(variables=['Political Affiliation'], evidence={'Economic Outlook': 0, 'Vote': 1})
# Positive economic outlook and liberal affiliation should lead to higher likelihood of vote 1.
print(result)
print("\n")
print(result_two)