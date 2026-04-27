from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Load your model
model = joblib.load('xgb_pipe.pkl')

# 2. Define the exact features your model expects and their rules
EXPECTED_FEATURES = {
    "order_status": {"type": str},
    "price": {"type": (int, float)},
    "freight_value": {"type": (int, float)},
    "product_name_lenght": {"type": (int, float)},
    "product_description_lenght": {"type": (int, float)},
    "product_photos_qty": {"type": (int, float)},
    "product_weight_g": {"type": (int, float)},
    "product_length_cm": {"type": (int, float)},
    "product_height_cm": {"type": (int, float)},
    "product_width_cm": {"type": (int, float)},
    "product_category_name": {"type": str},
    "prior_review_score_1": {"type": (int, float)},
    "prior_review_score_2": {"type": (int, float)},
    "payment_value": {"type": (int, float)},
    "payment_installments": {"type": (int, float)},
    "payment_type": {"type": str},
    "customer_lat": {"type": (int, float)},
    "customer_lng": {"type": (int, float)},
    "seller_lat": {"type": (int, float)},
    "seller_lng": {"type": (int, float)},
    "delivery_days": {"type": (int, float)},
    "delivery_days_precise": {"type": (int, float)},
    "delivery_vs_estimated": {"type": (int, float)},
    "prev_order_count": {"type": (int, float)},
    "is_repeat_customer": {"type": (int, float)},
    "freight_ratio": {"type": (int, float)},
    "product_volume_liters": {"type": (int, float)},
    "late_shipping": {"type": (int, float)},
    "order_total_items": {"type": (int, float)},
    "customer_state_order_count": {"type": (int, float)},
    "cust_state_by_n_orders_binned": {"type": (int, float, str)},
    "seller_state_order_count": {"type": (int, float)},
    "seller_state_by_n_orders_binned": {"type": (int, float, str)},
    "shipping_dist": {"type": (int, float)},
    "prev_product_id_count": {"type": (int, float)},
    "product_id_prior_score": {"type": (int, float)},
    "prev_seller_id_count": {"type": (int, float)},
    "seller_id_prior_score": {"type": (int, float)},
    "purchase_to_approval": {"type": (int, float)},
    "purchase_to_carrier": {"type": (int, float)},
    "purchase_to_estimate": {"type": (int, float)},
    "approval_to_carrier": {"type": (int, float)},
    "approval_to_customer": {"type": (int, float)},
    "approval_to_estimate": {"type": (int, float)},
    "carrier_to_customer": {"type": (int, float)},
    "carrier_to_estimate": {"type": (int, float)},
    "prior_rev_comments": {"type": (int, float)}
}


# --- HELPER FUNCTION: Validation ---
def validate_record(record):
    """Checks a single dictionary against our rules. Returns an error dict if it fails, or None if valid."""
    for field, rules in EXPECTED_FEATURES.items():
        # Check 1: Missing Field
        if field not in record:
            return {"error": "Missing required field", "details": {field: "This field is required"}}

        val = record[field]

        # Check 2: Invalid Type
        if not isinstance(val, rules["type"]):
            # Boolean is a subclass of int in Python, so we explicitly block it if we expect numbers
            if type(val) == bool and rules["type"] in [(int, float), int, float]:
                return {"error": "Invalid type", "details": {field: f"Expected {rules['type']}, got boolean"}}
            return {"error": "Invalid type", "details": {field: f"Expected {rules['type']}, got {type(val).__name__}"}}

        # Check 3: Invalid Value (e.g., negative prices)
        if "min" in rules and val < rules["min"]:
            return {"error": "Invalid value", "details": {field: f"Must be >= {rules['min']}"}}

    return None  # Passed all checks!


# --- ENDPOINT 1: Health Check ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "loaded"}), 200


# --- ENDPOINT 2: Single Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Run validation
    error = validate_record(data)
    if error:
        return jsonify(error), 400

    try:
        # Convert to DataFrame, predict, and format response
        df = pd.DataFrame([data])
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])

        # Determine the text label based on the prediction (Assuming 1 = Negative Review)
        label_text = "negative" if pred == 1 else "positive"

        return jsonify({"prediction": pred, "probability": round(prob, 4), "label": label_text}), 200

    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


# --- ENDPOINT 3: Batch Prediction ---
@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    records = request.get_json()

    # Batch size limit
    if not isinstance(records, list) or len(records) > 100:
        return jsonify({"error": "Invalid input", "details": "Must provide a list of up to 100 records"}), 400

    # Validate every record in the batch before predicting
    for i, record in enumerate(records):
        error = validate_record(record)
        if error:
            error["details"]["row_index"] = i  # Tell them which row failed
            return jsonify(error), 400

    try:
        df = pd.DataFrame(records)
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        results = []
        for p, prob in zip(preds, probs):
            results.append({
                "prediction": int(p),
                "probability": round(float(prob), 4),
                "label": "negative" if int(p) == 1 else "positive"
            })

        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": "Batch prediction failed", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

