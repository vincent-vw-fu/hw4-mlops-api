import requests

BASE_URL = "http://127.0.0.1:5001"

# --- REALISTIC FAKE DATA FOR TESTING ---
valid_record = {
    "order_status": "delivered",
    "price": 149.90,
    "freight_value": 18.50,
    "product_name_lenght": 55,
    "product_description_lenght": 450,
    "product_photos_qty": 2,
    "product_weight_g": 850,
    "product_length_cm": 30,
    "product_height_cm": 15,
    "product_width_cm": 20,
    "product_category_name": "relogios_presentes",
    "prior_review_score_1": 5.0,
    "prior_review_score_2": 4.0,
    "payment_value": 168.40,
    "payment_installments": 3,
    "payment_type": "credit_card",
    "customer_lat": -23.5505,
    "customer_lng": -46.6333,
    "seller_lat": -22.9068,
    "seller_lng": -43.1729,
    "delivery_days": 8,
    "delivery_days_precise": 8.2,
    "delivery_vs_estimated": -4.0,
    "prev_order_count": 0,
    "is_repeat_customer": 0,
    "freight_ratio": 0.11,
    "product_volume_liters": 9.0,
    "late_shipping": 0,
    "order_total_items": 1,
    "customer_state_order_count": 1500,
    "cust_state_by_n_orders_binned": 3,
    "seller_state_order_count": 800,
    "seller_state_by_n_orders_binned": 2,
    "shipping_dist": 400.5,
    "prev_product_id_count": 10,
    "product_id_prior_score": 4.5,
    "prev_seller_id_count": 45,
    "seller_id_prior_score": 4.8,
    "purchase_to_approval": 0.5,
    "purchase_to_carrier": 1.5,
    "purchase_to_estimate": 12.0,
    "approval_to_carrier": 1.0,
    "approval_to_customer": 7.5,
    "approval_to_estimate": 11.5,
    "carrier_to_customer": 6.5,
    "carrier_to_estimate": 10.5,
    "prior_rev_comments": 1
}

print("Starting API Tests...\n" + "="*40)

# TEST 1: Health Check
print("Test 1: GET /health")
res = requests.get(f"{BASE_URL}/health")
print(f"Status: {res.status_code}")
print(f"Response: {res.json()}\n")

# TEST 2: Valid Single Prediction
print("Test 2: POST /predict (Valid)")
res = requests.post(f"{BASE_URL}/predict", json=valid_record)
print(f"Status: {res.status_code}")
print(f"Response: {res.json()}\n")

# TEST 3: Valid Batch Prediction (5 records)
print("Test 3: POST /predict/batch (5 Valid Records)")
batch_data = [valid_record] * 5
res = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
print(f"Status: {res.status_code}")
print(f"Response: {res.json()}\n")

# TEST 4: Missing Required Field
print("Test 4: POST /predict (Missing Field)")
missing_field_record = valid_record.copy()
del missing_field_record["price"] # Explicitly removing the 'price' feature
res = requests.post(f"{BASE_URL}/predict", json=missing_field_record)
print(f"Status: {res.status_code} (Should be 400)")
print(f"Response: {res.json()}\n")

# TEST 5: Invalid Type
print("Test 5: POST /predict (Invalid Type)")
invalid_type_record = valid_record.copy()
invalid_type_record["price"] = "one hundred dollars" # Intentionally breaking 'price'
res = requests.post(f"{BASE_URL}/predict", json=invalid_type_record)
print(f"Status: {res.status_code} (Should be 400)")
print(f"Response: {res.json()}\n")

print("="*40 + "\nAll tests completed.")