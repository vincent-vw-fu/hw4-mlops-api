# 1. Start with the lightweight Python image required by the rubric
FROM python:3.10-slim

# 2. Install the system-level library XGBoost needs to run on Linux
RUN apt-get update && apt-get install -y libgomp1

# 3. Set the working folder inside the virtual container
WORKDIR /app

# 4. Copy your requirements list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your local files (app.py, xgb_pipe.pkl, etc.) into the container
COPY . .

# 6. Expose Port 5000 for the API
EXPOSE 5000

# 7. Use gunicorn (production server) to run your app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]