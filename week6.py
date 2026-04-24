from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
import pandas as pd

# ---------------- DATA ----------------
data = pd.DataFrame({
    'Rain': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'TrafficJam': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
    'ArriveLate': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No']
})

# ---------------- MODEL ----------------
# Define Bayesian Network structure
model = DiscreteBayesianNetwork([
    ('Rain', 'TrafficJam'),
    ('TrafficJam', 'ArriveLate')
])

# Train model using Maximum Likelihood Estimation
model.fit(data)

# ---------------- OUTPUT CPDs ----------------
print("Conditional Probability Distributions (CPDs):\n")
for cpd in model.get_cpds():
    print(cpd)
    print()

# ---------------- INFERENCE ----------------
inference = VariableElimination(model)

# Query: Probability of ArriveLate given Rain = Yes
query_result = inference.query(
    variables=['ArriveLate'],
    evidence={'Rain': 'Yes'}
)

print("Inference Result:\n")
print(query_result)