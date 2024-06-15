
# run pip install pgmpy in terminal to isntall pgmpy library.

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the structure of the Bayesian network
model = BayesianNetwork([
    ('Political Affiliation', 'Vote'), 
    ('Economic Outlook', 'Vote'), 
    ('Social Issues', 'Vote'),
    ('Media Influence', 'Vote'),
])

# Define the conditional probability distribution (CPD) for each node
cpd_political_affiliation = TabularCPD(variable='Political Affiliation', variable_card=3, values=[[0.4], [0.35], [0.25]])

cpd_economic_outlook = TabularCPD(variable='Economic Outlook', variable_card=2, values=[[0.6], [0.4]])

cpd_social_issues = TabularCPD(variable='Social Issues', variable_card=2, values=[[0.5], [0.5]])

cpd_media_influence = TabularCPD(variable='Media Influence', variable_card=2, values=[[0.7], [0.3]])

cpd_vote = TabularCPD(variable='Vote', variable_card=2, 
                                  values=[
                                      [0.7, 0.6, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.7, 0.6, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.7, 0.6, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01],
                                      [0.3, 0.4, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.3, 0.4, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99],
                                  ], 
                                  evidence=['Political Affiliation', 'Economic Outlook', 'Social Issues', 'Media Influence'], 
                                  evidence_card=[3, 2, 2, 2])

# Add CPDs to the model
model.add_cpds(cpd_political_affiliation, cpd_economic_outlook, cpd_social_issues, cpd_media_influence, cpd_vote)

# Check if the model is valid
assert model.check_model()

# Perform inference
inference = VariableElimination(model)

# Query the model: P(Vote | Political Affiliation=0, Economic Outlook=1, Social Issues=0, Media Influence=1)
result = inference.query(variables=['Vote'], evidence={'Political Affiliation': 0, 'Economic Outlook': 1, 'Social Issues': 0, 'Media Influence': 1})
print(result)