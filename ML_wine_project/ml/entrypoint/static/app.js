// app.js — handles predict and ingest flows (posts underscore keys to API)
const fields = [
  {name: "fixed_acidity", label: "fixed acidity", placeholder: "3.5-12.0"},
  {name: "volatile_acidity", label: "volatile acidity", placeholder: "0.05-1.0"},
  {name: "citric_acid", label: "citric acid", placeholder: "0.0-1.0"},
  {name: "residual_sugar", label: "residual sugar", placeholder: "0.0-40.0"},
  {name: "chlorides", label: "chlorides", placeholder: "0.005-0.2"},
  {name: "free_sulfur_dioxide", label: "free sulfur dioxide", placeholder: "0.0-150.0"},
  {name: "total_sulfur_dioxide", label: "total sulfur dioxide", placeholder: "0.0-300.0"},
  {name: "density", label: "density", placeholder: "0.987-1.010"},
  {name: "pH", label: "pH", placeholder: "2.8-3.8"},
  {name: "sulphates", label: "sulphates", placeholder: "0.2-1.0"},
  {name: "alcohol", label: "alcohol", placeholder: "8.0-14.5"}
];

function createField(item) {
  const {name, label, placeholder} = item;
  const row = document.createElement('div');
  row.className = 'row';
  const lab = document.createElement('label');
  lab.textContent = label;
  const input = document.createElement('input');
  input.type = 'number';
  input.step = 'any';
  input.name = name;
  input.required = true;
  input.placeholder = placeholder;
  row.appendChild(lab);
  row.appendChild(input);
  return row;
}

async function predictFeatures(features) {
  // POST to /predict/value (single-row predict)
  const res = await fetch('/predict/value', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(features)
  });
  const json = await res.json();
  if (!res.ok) throw json;
  return json; // expects {status: "success", prediction: ..., ...}
}

async function ingest(data) {
  // POST to /ingest (must include 'quality' key)
  const res = await fetch('/ingest', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  });
  const json = await res.json();
  if (!res.ok) throw json;
  return json;
}

document.addEventListener('DOMContentLoaded', () => {
  const fieldsDiv = document.getElementById('fields');
  fields.forEach(f => fieldsDiv.appendChild(createField(f)));

  const form = document.getElementById('sampleForm');
  const predictBtn = document.getElementById('predictBtn');
  const submitBtn = document.getElementById('submitBtn');
  const sampleResult = document.getElementById('sampleResult');
  const predictResult = document.getElementById('predictResult');

  predictBtn.addEventListener('click', async () => {
    // collect features only
    const features = {};
    fields.forEach(f => {
      const el = form.elements[f.name];
      if (!el) return;
      const v = el.value;
      if (v === '') return;
      features[f.name] = parseFloat(v);
    });
    if (Object.keys(features).length === 0) {
      predictResult.textContent = 'Please fill at least one feature.';
      return;
    }
    predictResult.textContent = 'Predicting...';
    try {
      const j = await predictFeatures(features);
      // try to find the prediction value in response (common shapes)
      let pred = j.prediction ?? j.predicted ?? j.result ?? null;
      // if response includes nested payload, try common shapes
      if (pred === null && Array.isArray(j.predictions) && j.predictions.length > 0) {
        pred = j.predictions[0].prediction ?? j.predictions[0];
      }
      //pred = parseInt(pred)
      predictResult.textContent = 'Predicted quality: ' + (pred ?? JSON.stringify(j));
    } catch (err) {
      predictResult.textContent = 'Predict error: ' + (err.message || JSON.stringify(err));
    }
  });

  submitBtn.addEventListener('click', async () => {
    // collect features and quality
    const features = {};
    fields.forEach(f => {
      const el = form.elements[f.name];
      if (!el) return;
      const v = el.value;
      if (v === '') return;
      features[f.name] = parseFloat(v);
    });
    const qel = form.elements['quality'];
    const q = (qel && qel.value !== '') ? parseInt(qel.value, 10) : null;

    if (!q && q !== 0) {
      sampleResult.textContent = 'Please provide the actual quality before submitting.';
      return;
    }
    // show predicted value first (optional)
    predictResult.textContent = 'Predicting...';
    try {
      const j = await predictFeatures(features);
      let pred = j.prediction ?? j.predicted ?? null;
      if (pred === null && Array.isArray(j.predictions) && j.predictions.length > 0) {
        pred = j.predictions[0].prediction ?? j.predictions[0];
      }
      //pred = parseInt(pred)
      predictResult.textContent = 'Predicted quality: ' + (pred ?? JSON.stringify(j));
    } catch (err) {
      predictResult.textContent = 'Predict error: ' + (err.message || JSON.stringify(err));
      // continue: we still submit the sample even if predict failed
    }

    // Now append quality and ingest
    const data = {...features, quality: q};
    sampleResult.textContent = 'Submitting sample...';
    try {
      const j2 = await ingest(data);
      sampleResult.textContent = 'Ingest success: ' + (j2.message || JSON.stringify(j2));
    } catch (err) {
      sampleResult.textContent = 'Ingest error: ' + (err.message || JSON.stringify(err));
    }
  });

  // retrain button left unchanged
  const retrainBtn = document.getElementById('retrainBtn');
  retrainBtn.addEventListener('click', async () => {
    const div = document.getElementById('retrainResult');
    div.textContent = 'Retraining... (this may take a while)';
    try {
      const r = await fetch('/retrain', {method: 'POST', headers:{'Content-Type':'application/json'}, body: "{}"});
      const j = await r.json();
      if (r.ok) div.textContent = 'Retrain success: ' + (j.message || 'OK');
      else div.textContent = 'Retrain error: ' + (j.message || JSON.stringify(j));
    } catch (err) {
      div.textContent = 'Network error: ' + err;
    }
  });
});
