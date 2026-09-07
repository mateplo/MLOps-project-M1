/* In-browser inference for the Adult Income Classifier.
 * config.json (written by the CI) says which HF model repo / revision to load.
 * preprocess.json (exported with the ONNX model) says how to prepare the inputs. */
(async function () {
  const $ = (id) => document.getElementById(id);
  const DEFAULTS = {
    age: 42, education_num: 13, hours_per_week: 40, capital_gain: 5178, capital_loss: 0,
    workclass: "Private", marital_status: "Married-civ-spouse", occupation: "Exec-managerial",
    relationship: "Husband", race: "White", sex: "Male", native_country: "United-States",
  };
  const NUMERIC_HINTS = { age: [17, 90], education_num: [1, 16], hours_per_week: [1, 99], capital_gain: [0, 99999], capital_loss: [0, 4356] };

  function fail(msg, err) {
    $("status").textContent = msg;
    $("error").textContent = err ? String(err.stack || err) : "";
    console.error(msg, err);
  }

  let cfg;
  try {
    cfg = await (await fetch("./config.json", { cache: "no-store" })).json();
  } catch (e) {
    cfg = { model_repo: "Mateplo/adult-income-classifier", model_revision: "main" };
  }
  const base = cfg.base_url || `https://huggingface.co/${cfg.model_repo}/resolve/${cfg.model_revision}/`;

  let spec, session;
  try {
    $("status").textContent = `Downloading model from ${cfg.model_repo}@${cfg.model_revision}…`;
    const [specResp, onnxResp] = await Promise.all([fetch(base + "preprocess.json"), fetch(base + "model.onnx")]);
    if (!specResp.ok || !onnxResp.ok) throw new Error(`HTTP ${specResp.status} / ${onnxResp.status} from ${base}`);
    spec = await specResp.json();
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
    session = await ort.InferenceSession.create(await onnxResp.arrayBuffer(), { executionProviders: ["wasm"] });
  } catch (e) {
    return fail("Could not load the model.", e);
  }

  // ---- identity -------------------------------------------------------------
  const meta = $("meta");
  for (const [k, v] of [
    ["model", `${spec.model_name} v${spec.version} (${spec.model_type})`],
    ["run_id", spec.run_id],
    ["source", `${cfg.model_repo}@${cfg.model_revision}`],
    ["threshold", Number(spec.decision_threshold).toFixed(4)],
    ["runtime", `onnxruntime-web ${ort.env.versions?.web || ""} (wasm)`],
  ]) {
    meta.insertAdjacentHTML("beforeend", `<dt>${k}</dt><dd>${v ?? "—"}</dd>`);
  }

  // ---- form ---------------------------------------------------------------
  const form = $("form");
  for (const c of spec.numeric) {
    const [min, max] = NUMERIC_HINTS[c] || [0, 1e6];
    form.insertAdjacentHTML("beforeend",
      `<label>${c}<input name="${c}" type="number" min="${min}" max="${max}" step="1" value="${DEFAULTS[c] ?? min}"></label>`);
  }
  for (const c of spec.categorical) {
    const opts = spec.frequent_categories[c].map((v) => `<option ${v === DEFAULTS[c] ? "selected" : ""}>${v}</option>`).join("");
    form.insertAdjacentHTML("beforeend",
      `<label>${c}<select name="${c}">${opts}<option value="${spec.infrequent_token}">other (rare)</option><option value="${spec.missing_token}">unknown</option></select></label>`);
  }
  form.insertAdjacentHTML("beforeend", `<button type="submit">Predict</button>`);

  // ---- preprocessing mirrors src/export_onnx.py::browser_preprocess ------------
  function buildFeeds(values) {
    const feeds = {};
    for (const c of spec.numeric) {
      const x = parseFloat(values[c]);
      feeds[c] = new ort.Tensor("float32", Float32Array.from([Number.isFinite(x) ? x : NaN]), [1, 1]);
    }
    for (const c of spec.categorical) {
      let v = (values[c] ?? "").toString().trim();
      if (v === "") v = spec.missing_token;
      const allowed = new Set([...spec.frequent_categories[c], spec.missing_token, spec.infrequent_token]);
      if (!allowed.has(v)) v = spec.infrequent_token;
      feeds[c] = new ort.Tensor("string", [v], [1, 1]);
    }
    return feeds;
  }

  async function predict() {
    const values = Object.fromEntries(new FormData(form).entries());
    const out = await session.run(buildFeeds(values));
    const probs = out[session.outputNames[1]].data; // zipmap=False -> [n, 2]
    const score = Number(probs[1]);
    const thr = Number(spec.decision_threshold);
    const positive = score >= thr;
    $("result").hidden = false;
    $("label").textContent = positive ? spec.labels["1"] : spec.labels["0"];
    $("label").className = "label " + (positive ? "pos" : "neg");
    $("fill").style.width = `${(score * 100).toFixed(1)}%`;
    $("thr").style.left = `${(thr * 100).toFixed(1)}%`;
    $("score").textContent = `P(>50K) = ${score.toFixed(4)}`;
    $("threshold").textContent = thr.toFixed(4);
    $("status").textContent = "Ready.";
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = form.querySelector("button"); btn.disabled = true;
    try { await predict(); } catch (e) { fail("Prediction failed.", e); } finally { btn.disabled = false; }
  });
  $("status").textContent = "Ready.";
  await predict();
})();
