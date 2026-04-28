# Olist E-Commerce Satisfaction Predictor API
**Author:** Vincent Vojack-Weeks | DATA 6545 HW4 

## Project Overview
This repository contains a production-ready Machine Learning API that predicts customer satisfaction for the Olist dataset. The deployed model is an 
- XGBoost Classifier 
- wrapped in a Flask API
- containerized using Docker, and 
- deployed to the cloud using Render

It features data validation, batch prediction capabilities, and drift monitoring.

## Live URL
The API is currently deployed and live on Render at:
**[https://hw4-mlops-api-82de.onrender.com](https://hw4-mlops-api-82de.onrender.com)**

---

## API Documentation

### `GET /health`
Checks if the API is running and the model is loaded in memory.
* **Request:** `curl -X GET https://hw4-mlops-api-82de.onrender.com/health`
* **Response:**
    ```json
    {
      "status": "healthy",
      "model": "loaded"
    }
    ```

### `POST /predict`
Generates a prediction for a single Olist order.
* **Request Body:** JSON object containing the 47 required features.
* **Response:**
    ```json
    {
      "prediction": 0,
      "label": "positive",
      "probability": 0.0281
    }
    ```

### `POST /predict/batch`
Generates predictions for multiple orders simultaneously.
* **Request Body:** A JSON array of order objects `[ {order_1}, {order_2} ]`.
* **Response:**
    ```json
    {
      "results": [
        {"prediction": 0, "label": "positive", "probability": 0.0281},
        {"prediction": 1, "label": "negative", "probability": 0.8912}
      ]
    }
    ```

---

## Input Schema
The API enforces input validation. It expects the following 47 features. Missing fields or incorrect data types will return an `HTTP 400 Bad Request`.

| Feature Name | Data Type | Description / Valid Values |
| :--- | :--- | :--- |
| `order_status` | String | e.g., 'delivered', 'shipped' |
| `payment_type` | String | e.g., 'credit_card', 'boleto' |
| `product_category_name` | String | e.g., 'relogios_presentes', 'electronics' |
| `price`, `freight_value`, `payment_value` | Float/Int | Currency values > 0 |
| `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm`, `product_volume_liters` | Float/Int | Physical dimensions |
| `product_name_lenght`, `product_description_lenght`, `product_photos_qty` | Int | Listing details |
| `prior_review_score_1`, `prior_review_score_2`, `product_id_prior_score`, `seller_id_prior_score` | Float | 1.0 to 5.0 |
| `customer_lat`, `customer_lng`, `seller_lat`, `seller_lng` | Float | Geographic coordinates |
| `delivery_days`, `delivery_days_precise`, `delivery_vs_estimated` | Float/Int | Transit metrics |
| `purchase_to_approval`, `purchase_to_carrier`, `purchase_to_estimate`, `approval_to_carrier`, `approval_to_customer`, `approval_to_estimate`, `carrier_to_customer`, `carrier_to_estimate` | Float | Logistic timeframes (days) |
| `prev_order_count`, `is_repeat_customer`, `order_total_items` | Int | Customer/Order history |
| `customer_state_order_count`, `seller_state_order_count`, `cust_state_by_n_orders_binned`, `seller_state_by_n_orders_binned` | Int/String | State-level volume metrics |
| `prev_product_id_count`, `prev_seller_id_count`, `prior_rev_comments` | Int | Historical aggregates |
| `freight_ratio`, `shipping_dist` | Float | Shipping cost/distance ratios |
| `late_shipping` | Int | 0 (No) or 1 (Yes) |

---

## Local Setup

### Without Docker
1. Clone the repository and navigate to the directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Flask server:
   ```bash
   python3 app.py
   ```
4. Open a second terminal tab and run the test script to verify the API:
   ```bash
   python3 test_api.py
   ```

### Running with Docker
1. Build the Docker image using the provided Dockerfile:
   ```bash
   docker build -t hw4-api .
   ```
2. Run the container, mapping the container's port 5000 to your local port 5000:
   ```bash
   docker run -p 5000:5000 hw4-api
   ```
3. Open a second terminal tab and run the test script to verify the API is responding from within the container:
   ```bash
   python3 test_api.py
   ```
   
## Model Information

* **Deployed Model:** XGBoost Classifier
* **Key Performance Metrics (Baseline):**
  * **Accuracy:** ~0.84
  * **F1 Score:** ~0.59
  * **AUC:** ~0.78
* **Known Limitations & Monitoring Requirements:**
  * **Limited Date Data:** All the data for this model is based on orders from 2016-2018
  * **Maintenance Trigger:** Due to potential sensitivity to data drift, the model requires continuous monitoring. It is recommended to trigger an automated retraining pipeline if the Population Stability Index (PSI) for any core feature exceeds the 0.15 threshold.
