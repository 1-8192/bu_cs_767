# run pip install pgmpy in terminal to install pgmpy library.

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian network
model = BayesianNetwork([
    ('Economic Outlook', 'Political Affiliation'), 
    ('Political Affiliation', 'Vote'), 
    ('Economic Outlook', 'Vote')
])

# Define the conditional probability distribution (CPD) for each node
# slightly higher chance of negative economic outlook.
cpd_economic_outlook = TabularCPD(variable='Economic Outlook', variable_card=2, values=[[0.6], [0.4]])

# Assuming 0: Negative, 1: Positive for Economic Outlook
# Assuming 0: Conservative, 1: Liberal, 2: Other for Political Affiliation
cpd_political_affiliation = TabularCPD(variable='Political Affiliation', variable_card=3, 
                                       values=[
                                           [0.6, 0.2],  # Higher probability of being conservative if economic outlook is negative
                                           [0.3, 0.5],  # Higher probability of being liberal if economic outlook is positive
                                           [0.1, 0.3]   # Other remains less likely in both cases
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

# Query the model: P(Vote | Economic Outlook=0, Political Affiliation=0)
result = inference.query(variables=['Vote'], evidence={'Economic Outlook': 0, 'Political Affiliation': 0})
print(result)