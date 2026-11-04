import pickle                                              #ex7
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)

with open("model.pkl","rb") as file:
  model = pickle.load(file)


@app.route("/")
def home():
  return jsonify({"message": "App is running..."})

@app.route("/health")
def health():
  return jsonify({"message": "App is healthy..."})

@app.route("/predict",methods=["POST"])
def predict():
  data = request.get_json()
  s_l = data["sepal_length"]
  s_w = data["sepal_width"]
  p_l = data["petal_length"]
  p_w = data["petal_width"]
  
  prediction = model.predict([[s_l, s_w, p_l, p_w]])
  return jsonify({"prediction": prediction[0]})               # model always return a list so we taking the first element of the list and returning it as a response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
 









#import pickle   #ex6

#with open("model.pkl","rb") as file:
    #model = pickle.load(file)

#samples=[[1.2, 1.4, 0.3, 2.5]]

# s_l = float(input("Enter Sepal Length: "))
# s_w = float(input("Enter Sepal Width: "))
# p_l = float(input("Enter Petal Length: "))
# p_w = float(input("Enter Petal Width: "))

#pred = model.predict(samples)

#print(pred)


# pred = model.predict([[s_l, s_w, p_l, p_w]])
#print(pred)