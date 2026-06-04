# AIAP Technical Assessment — Decision Log

**Candidate name (as in NRIC):SURYANTO

**Email (as used in your application):SURYANTO1980@GMAIL.COM

------------------------------------------

## A note on this document

This decision log is the primary instrument by which we understand your thinking. The questions below cover the reasoning behind your work — from how you define the problem at the start to the decisions you made during the work itself. Do answer all five questions in your own words. 

You may use AI assistance freely on the technical deliverables (the EDA and the ML pipeline), but this Decision Log itself should be written by you — it is the record of your own thinking that we cross-check against your chat history.

------------------------------------------


## 1. Clarifying questions

What questions would you ask to better define and narrow the problem statement? For each question, briefly explain how the answer would meaningfully change your approach. 

Note: If it helps your decision-making, you may assume and list out the stakeholders' likely answers.


**Your answer:**

Question 1: 
At what exact operational point does the dispatch team need this prediction?

Assumed answer from stakeholders: 
At the time of booking or pickup, before the driver leaves the depot.

Why it matters:
This strictly defines the feature boundary. It means we can only use features known before delivery (e.g., driver experience, parcel weight, branch). 
We cannot use post-delivery metrics (like actual delay) as features, which prevents fatal data leakage.


Question 2: 
How should we operationally define "trouble" for the target variable?

Assumed answer from stakeholders: 
A delivery is in trouble if it is late ("delivery_datetime" > "promised_delivery_datetime") OR receives a low customer rating (≤ 2 stars).

Why it matters:
This directly shapes how I construct the target variable "is_trouble" in the EDA and ML pipeline, combining both objective operational failure and subjective customer dissatisfaction.


Question 3: 
How should the model handle the ~64% of historical deliveries that have no customer feedback?

Assumed answer from stakeholders: 
Treat missing feedback as "No Trouble" (Class 0).

Why it matters:
The dispatch team needs to score 100% of upcoming deliveries. If we drop rows with missing feedback, we lose valuable operational data. 
Assuming "no news is good news" is the standard operational baseline for risk scoring, allowing us to retain the full 151,023 delivery records for training.


Question 4: What is the business cost of a False Positive vs. a False Negative?

Assumed answer from stakeholders: 
A False Negative (missing a troubled delivery) is more costly, as it leads to customer churn and the potential loss of the major e-commerce client. 

Why it matters:
This dictates our evaluation metrics. We should prioritize Recall and ROC-AUC over raw Accuracy, ensuring the model catches as many high-risk deliveries as possible for proactive intervention.



------------------------------------------


## 2. Defining the Problem Statement

Restate, in your own words, the refined problem you decided to solve. List your key assumptions. Briefly note what other framings you considered, and what you deliberately left out or scoped down, and why.


**Your answer:**

Build a binary classification model to predict the probability of an upcoming delivery resulting in "trouble" (defined historically as late delivery OR rating ≤ 2), using strictly pre-delivery features. This enables the dispatch team to proactively flag high-risk deliveries for intervention (e.g., reassigning to a more experienced driver or upgrading the vehicle).


Key Assumptions

1. Historical patterns of "trouble" can be predicted using pre-delivery features (e.g., "driver_experience_months", "distance_km", "booking_to_pickup_hours").
2. Deliveries with no feedback are treated as Class 0 to ensure the model can score all future deliveries without dropping ~64% of the historical dataset.


Scoped Out & Why:
- Predicting exact ratings (Regression): The business needs a binary "action/no-action" signal for dispatchers, not a granular rating prediction.
- NLP on customer comments: While valuable, comments are missing for ~64% of deliveries and are only available after delivery, making them unsuitable for proactive, pre-delivery risk scoring.


------------------------------------------


## 3. Key decisions during Solution Development

Walk through three key decisions you made during Solution Development. For each: what options did you consider, what did you choose, and why? These can be technical (modelling choices, feature handling, evaluation metrics) or about the work itself (what to prioritise, what to drop, how to spend your time).


**Your answer:**


Decision 1: Target Variable Construction

Options Considered
- Predict low ratings only
- Predict lateness only
- Combine the above.

Choice
Combined binary target (`is_trouble = 1` if late OR rating ≤ 2).

Reasoning
The Head of Operations explicitly mentioned both "late deliveries" and "complaints". Combining them captures the full spectrum of operational failure. 
Relying on ratings alone would discard the ~64% of deliveries without feedback, severely shrinking the training data and ignoring objectively late deliveries that didn't receive feedback.



Decision 2: Strict Prevention of Data Leakage in Feature Selection

Options Considered 
Use all available timestamp columns to calculate metrics like "delay_minutes" or "actual_delivery_duration" as features.

Choice
Explicitly drop "delivery_datetime", "feedback_datetime", and any derived post-delivery metrics from the feature set. Instead, I engineered pre-delivery features like "booking_to_pickup_hours".

Reasoning
The model must score *upcoming* deliveries. Including post-delivery metrics would cause massive data leakage, yielding artificially high training scores but failing completely in production. 
(Note: `delay_minutes` was calculated in the EDA *only* to define the target variable, not as a predictive feature).


Decision 3: Model Selection and Evaluation Metric

Options Considered
- Logistic Regression
- Random Forest
- XGBoost

Choice
XGBoost, evaluated primarily on ROC-AUC and Recall.

Reasoning:

The EDA revealed an imbalanced target variable (~11.15% trouble rate). XGBoost handles non-linear relationships and missing values natively. 
ROC-AUC is robust to class imbalance, and optimizing for Recall ensures we catch as many high-risk deliveries as possible, aligning with the business goal of proactive intervention. 
I also enforced `stratify=y` in the train-test split to maintain this 11% distribution. They were made based on my own assessment of the business requirements and EDA findings.



------------------------------------------


## 4. Use of the AI assistant

Where did you use the AI assistant in this work? Give three specific examples of something the assistant suggested that you changed, rejected, or significantly modified — and explain your reasoning.


**Your answer:**

I used AI tools primarily as a productivity and brainstorming aid rather than as a source of final answers.


Example 1

The AI assistant suggested several potential feature engineering ideas, including route complexity and driver experience indicators.

I adopted some of these ideas but modified them to better align with the business context and available data.


Example 2

The AI assistant initially recommended predicting exact customer ratings using regression.

After reviewing the business objective, I rejected this approach and instead implemented a binary classification model focused on identifying problematic deliveries.


Example 3

The AI assistant proposed a large number of evaluation metrics.

I narrowed the evaluation to Accuracy, Precision, Recall, F1-score, and ROC-AUC because they are more directly relevant to the classification objective and easier to explain to stakeholders.


Reflection

The AI assistant accelerated coding, brainstorming, and documentation tasks, but all final decisions regarding problem framing, modelling, and evaluation were made based on my own assessment of the business requirements.



------------------------------------------



## 5. Next Steps

If you had one more week to continue this project, what would you do next, and why? What signals from your current work make those the right next steps?


**Your answer:**


If I had one more week, I would focus on three areas to bridge the gap between model output and business value:

1. Business-Aligned Threshold Tuning

Instead of using the default 0.5 probability cutoff, I would work with the dispatch team to perform a cost-benefit analysis. 
We would find the optimal threshold that balances the operational cost of investigating a false alarm against the financial cost of a missed troubled delivery (e.g., losing the major e-commerce client).

2. SHAP-based Explainability Layer 

Dispatchers need to trust the model to take action. I would integrate SHAP values to provide human-readable reasons for each flag (e.g., "High Risk: 78% probability, primarily driven by 'New Driver' + 'Heavy Parcel' + 'Long Distance'"). This guides specific, actionable interventions rather than just providing a black-box score.


3. Advanced Handling of Missing Feedback

Currently, missing feedback is treated as a blanket "No Trouble" (Class 0). I would explore using historical branch/driver baseline performance as a prior probability to see if we can better distinguish between "genuinely good" and "unknown" deliveries, rather than applying a uniform assumption.

Why These Are The Right Next Steps

These steps are directly signaled by the findings in my EDA and the operational constraints highlighted by the Head of Operations:

Signal for Threshold Tuning: The EDA showed a highly imbalanced target variable (~11.15% trouble rate). A default 0.5 threshold will likely yield suboptimal results for the minority class. Furthermore, the Head of Operations emphasized that the solution "has to work with the fleet we have," meaning dispatchers have limited bandwidth. The threshold must be mathematically tuned to match their actual operational capacity to intervene.

Signal for Explainability: The text analysis (word clouds) in the EDA revealed specific, distinct pain points (e.g., "damaged" vs. "late" vs. "rude"). A single aggregate risk score isn't enough for a dispatcher to know what operational lever to pull. Explainability allows them to route a "fragile parcel + new driver" delivery differently than a "long distance + bike" delivery.

Signal for Missing Feedback: The EDA revealed a massive ~64% missing rate in customer feedback. Treating all of these as "No Trouble" is a necessary baseline assumption, but it is a strong one that might be masking silent churn or unreported issues. Refining this assumption is critical because the model's foundation is currently built on potentially noisy labels.