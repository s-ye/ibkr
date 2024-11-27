# Probabilistic Forecasting for Future Prices using NLL

## Overview
This document explains the rationale for using a neural network to predict the **probability distribution** for future log returns of stock prices and describes the role of the **Negative Log-Likelihood (NLL)** loss function in this setup.

---

## Problem Statement
Traditional price forecasting models typically provide **point predictions**, such as the expected price or return for a future time step. While these methods are useful, they fail to account for the **uncertainty** inherent in financial markets. In this implementation:
- Instead of a point forecast, the neural network predicts a **probabilistic distribution** for future log returns.
- Specifically, the model predicts the **mean** (\(\mu_t\)) and **standard deviation** (\(\sigma_t\)) of a Gaussian distribution at each time step.

This approach provides both:
1. An estimate of the **expected behavior** (\(\mu_t\)).
2. A measure of **uncertainty** or risk (\(\sigma_t\)).

---

## Why Predict a Distribution?
1. **Uncertainty Quantification**:
    - Financial markets are inherently volatile, with price movements subject to randomness and noise.
    - By predicting a probability distribution, we capture both the central tendency (\(\mu_t\)) and the variability (\(\sigma_t\)) of future log returns.

2. **Risk-Aware Decision-Making**:
    - Knowing the uncertainty (\(\sigma_t\)) allows us to make decisions based on confidence intervals or probabilistic thresholds.
    - For example:
      - **Buy**: When the lower bound of the confidence interval (\(\mu_t - 2\sigma_t\)) exceeds a certain threshold.
      - **Avoid**: When the uncertainty (\(\sigma_t\)) is too high.

3. **More Realistic Modeling**:
    - Log returns often follow an approximate Gaussian distribution, especially over short time intervals.
    - Modeling log returns as samples from \(\mathcal{N}(\mu_t, \sigma_t^2)\) aligns with real-world financial behavior.

---

## What the Neural Network Learns
The neural network predicts:
- **\(\mu_t\)**: The mean of the Gaussian distribution, representing the expected future log return.
- **\(\sigma_t\)**: The standard deviation of the Gaussian distribution, representing the uncertainty or risk.

At each future time step \( t \) in the forecast horizon, the predicted log return (\(y_t\)) is treated as a sample from this distribution:

\[
y_t \sim \mathcal{N}(\mu_t, \sigma_t^2)
\]

---

## The Role of Negative Log-Likelihood (NLL) Loss

The **Negative Log-Likelihood (NLL) Loss** measures how well the predicted distribution \(\mathcal{N}(\mu_t, \sigma_t^2)\) aligns with the actual observed data. For a single time step \( t \), the likelihood of observing the true value \( y_t \) under the predicted Gaussian is:

\[
p(y_t | \mu_t, \sigma_t) = \frac{1}{\sqrt{2\pi \sigma_t^2}} \exp\left(-\frac{(y_t - \mu_t)^2}{2\sigma_t^2}\right)
\]

Taking the logarithm gives the **log-likelihood**:

\[
\log p(y_t | \mu_t, \sigma_t) = -\frac{1}{2} \log(2\pi) - \log(\sigma_t) - \frac{(y_t - \mu_t)^2}{2\sigma_t^2}
\]

The **Negative Log-Likelihood (NLL)** is the negative of this term (ignoring constants):

\[
\text{NLL} = \log(\sigma_t) + \frac{(y_t - \mu_t)^2}{2\sigma_t^2}
\]

### Key Components of NLL
1. **Uncertainty Penalization (\(\log(\sigma_t)\))**:
    - Penalizes large uncertainty (\(\sigma_t\)).
    - Encourages the model to predict smaller \(\sigma_t\) values when it is confident in its prediction.

2. **Error Penalization (\(\frac{(y_t - \mu_t)^2}{2\sigma_t^2}\))**:
    - Penalizes predictions where the true value (\(y_t\)) is far from the mean (\(\mu_t\)).
    - Scaled inversely by \(\sigma_t^2\), so larger uncertainty allows for more tolerance in errors.

For a sequence of future time steps (\(t = 1, 2, \dots, H\)), the total NLL loss is the sum across all steps:

\[
\text{Total NLL Loss} = \sum_{t=1}^H \left[ \log(\sigma_t) + \frac{(y_t - \mu_t)^2}{2\sigma_t^2} \right]
\]

---

## Why NLL is Suitable for This Problem
1. **Probabilistic Predictions**:
    - NLL encourages the model to output a distribution (\(\mu_t, \sigma_t\)) that accurately reflects the observed data.

2. **Uncertainty Awareness**:
    - The model learns to balance accuracy (\(y_t \approx \mu_t\)) with realistic uncertainty (\(\sigma_t\)).

3. **Alignment with Data**:
    - Log returns are approximately Gaussian-distributed, making NLL a natural choice.

---

## Limitations of This Approach
1. **Gaussian Assumption**:
    - The model assumes that log returns follow a Gaussian distribution, which may not hold during extreme market events.

2. **Independence of Time Steps**:
    - The predicted distributions for different time steps are independent, ignoring potential correlations.

3. **Limited Features**:
    - This model uses only log returns and volume as inputs. Adding more features (e.g., technical indicators or macroeconomic data) could improve performance.

---

## Conclusion
Using a neural network to predict a probability distribution for future log returns provides both **expected values** and **uncertainty estimates**, enabling robust and risk-aware decision-making. The **Negative Log-Likelihood Loss** ensures that the predicted distributions align with observed data while balancing accuracy and uncertainty.

This probabilistic approach aligns with real-world financial behavior and opens the door to applications in:
- Risk management.
- Scenario analysis.
- Probabilistic trading strategies.