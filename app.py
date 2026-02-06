from flask import Flask, request, render_template_string, jsonify
from predict import predict_urls
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# HTML page
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Phishing URL Checker</title>
  <style>
    body { font-family: Arial, sans-serif; padding: 20px; }
    .safe { color: green; font-weight: bold; }
    .phishing { color: red; font-weight: bold; }
  </style>
</head>
<body>
  <h1>Phishing URL Checker</h1>
  <textarea id="urls" rows="5" cols="50" placeholder="Enter one URL per line"></textarea><br><br>
  <button onclick="checkUrls()">Check URLs</button>
  <h2>Results:</h2>
  <div id="results"></div>

  <script>
    async function checkUrls() {
      let urls = document.getElementById("urls").value
        .split("\\n")
        .map(u => u.trim())
        .filter(u => u !== "")
        .map(u => u.startsWith("http") ? u : "http://" + u); // normalize

      const resContainer = document.getElementById("results");
      resContainer.innerHTML = "Checking...";

      try {
        const response = await fetch("/check", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ urls })
        });

        const data = await response.json();
        resContainer.innerHTML = "";

        if (data.error) {
          resContainer.textContent = "Error: " + data.error;
          return;
        }

        if (data.length === 0) {
          resContainer.textContent = "No results.";
          return;
        }

        data.forEach(item => {
          const div = document.createElement("div");

          // Decide safe/phishing based on probability > 50%
          const label = item.prob > 50 ? "PHISHING" : "SAFE";
          div.textContent = `${item.url} → ${label} (${item.prob.toFixed(2)}%)`;
          div.className = label === "SAFE" ? "safe" : "phishing";

          resContainer.appendChild(div);
        });
      } catch (err) {
        resContainer.textContent = "Error: " + err;
      }
    }
  </script>
</body>
</html>
"""

# Serve the HTML page
@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

# API endpoint
@app.route('/check', methods=['POST'])
def check_url():
    data = request.get_json()
    urls = data.get('urls', [])
    results = []

    if urls:
        try:
            preds = predict_urls(urls)  # [(url, pred, prob), ...]
            for u, p, prob in preds:
                # Convert float32 to Python float and make percentage
                prob_percent = float(prob) * 100
                results.append({
                    "url": u,
                    "prob": prob_percent
                })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
