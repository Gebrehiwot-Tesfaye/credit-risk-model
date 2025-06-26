# credit-risk-model

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord requires financial institutions to rigorously measure, manage, and report credit risk. This regulatory framework emphasizes transparency, accountability, and the ability to justify risk assessments to both regulators and stakeholders. As a result, our credit scoring model must be interpretable and well-documented, enabling clear explanations of how risk scores are derived and ensuring that decisions can be audited and defended. Interpretable models also help build trust with customers and regulators, and facilitate compliance with evolving regulatory standards.

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In the absence of a direct default label, we must engineer a proxy variable—such as behavioral patterns (e.g., Recency, Frequency, Monetary value) that correlate with credit risk—to categorize users as high or low risk. This proxy enables us to train and validate our model. However, relying on a proxy introduces risks: if the proxy does not accurately reflect true default behavior, the model may misclassify customers, leading to financial losses (by approving risky loans) or missed business opportunities (by rejecting creditworthy applicants). It is crucial to continuously validate and refine the proxy to minimize these risks.

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

Simple models, such as Logistic Regression with Weight of Evidence (WoE), offer high interpretability, making it easier to explain decisions to regulators and stakeholders, and to identify and mitigate bias. However, they may not capture complex patterns in the data, potentially limiting predictive performance. Complex models, like Gradient Boosting, can achieve higher accuracy by modeling nonlinear relationships, but they are less transparent and harder to interpret. In a regulated context, the trade-off is between maximizing predictive power and ensuring the model's decisions are explainable, auditable, and compliant with regulatory requirements. Often, a balance is sought by starting with interpretable models and only adopting more complex approaches if they provide substantial, justifiable improvements.
