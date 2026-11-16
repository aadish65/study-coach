"use strict";

require("dotenv").config();

const express = require("express");
const cors = require("cors");

// ---------------------------------------------------------------------------
// Configuration (override via backend/.env)
// ---------------------------------------------------------------------------
const PORT = Number(process.env.PORT) || 5000;
const FASTAPI_URL = (process.env.FASTAPI_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS) || 15000;
const MAX_STATEMENT_LENGTH = Number(process.env.MAX_STATEMENT_LENGTH) || 5000;

// Extra allowed origins, comma-separated (e.g. https://myapp.example.com)
const EXTRA_ORIGINS = (process.env.ALLOWED_ORIGINS || "")
  .split(",")
  .map((o) => o.trim())
  .filter(Boolean);

// ---------------------------------------------------------------------------
// App + middleware
// ---------------------------------------------------------------------------
const app = express();

// Local-development CORS:
//  - no Origin header (curl, Postman, server-to-server)
//  - "null" origin (index.html opened directly via file://)
//  - any http://localhost:* or http://127.0.0.1:* (VS Code Live Server, Vite, etc.)
//  - anything listed in ALLOWED_ORIGINS
const LOCAL_ORIGIN = /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/;

app.use(
  cors({
    origin(origin, callback) {
      if (!origin || origin === "null" || LOCAL_ORIGIN.test(origin) || EXTRA_ORIGINS.includes(origin)) {
        return callback(null, true);
      }
      return callback(new Error(`Origin not allowed by CORS: ${origin}`));
    },
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type"],
  })
);

app.use(express.json({ limit: "100kb" }));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** fetch() with a hard timeout. */
async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

/** Convert a fetch() failure into an HTTP status + message for the frontend. */
function describeUpstreamError(err) {
  if (err.name === "AbortError") {
    return {
      status: 504,
      error: "The ML service took too long to respond.",
      code: "ML_SERVICE_TIMEOUT",
    };
  }
  const cause = err.cause && err.cause.code ? err.cause.code : err.code;
  if (cause === "ECONNREFUSED" || cause === "ENOTFOUND" || cause === "ECONNRESET" || cause === "EAI_AGAIN") {
    return {
      status: 503,
      error: `Cannot reach the ML service at ${FASTAPI_URL}. Make sure FastAPI is running.`,
      code: "ML_SERVICE_UNAVAILABLE",
    };
  }
  return {
    status: 502,
    error: "Unexpected error while contacting the ML service.",
    code: "ML_SERVICE_ERROR",
  };
}

/** Pull a readable message out of a FastAPI error body (string or validation array). */
function extractDetail(body) {
  if (!body) return null;
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) {
    return body.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return null;
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

// GET /api/health -> status of this server AND the FastAPI service
app.get("/api/health", async (req, res) => {
  const result = {
    status: "ok",
    backend: "ok",
    mlService: "unknown",
    mlServiceUrl: FASTAPI_URL,
    timestamp: new Date().toISOString(),
  };

  try {
    const upstream = await fetchWithTimeout(`${FASTAPI_URL}/health`, {}, 5000);
    if (upstream.ok) {
      result.mlService = "ok";
      return res.status(200).json(result);
    }
    result.mlService = `error (HTTP ${upstream.status})`;
  } catch (err) {
    result.mlService = "unreachable";
    result.detail = describeUpstreamError(err).error;
  }

  result.status = "degraded";
  return res.status(503).json(result);
});

// POST /api/predict  { "statement": "..." }
app.post("/api/predict", async (req, res, next) => {
  try {
    const body = req.body;

    // --- Validation ---
    if (!body || typeof body !== "object" || Array.isArray(body)) {
      return res.status(400).json({
        error: 'Request body must be a JSON object like {"statement": "..."}',
        code: "INVALID_BODY",
      });
    }

    const { statement } = body;

    if (typeof statement !== "string") {
      return res.status(400).json({
        error: '"statement" is required and must be a string.',
        code: "INVALID_STATEMENT",
      });
    }

    const cleaned = statement.trim();

    if (cleaned.length === 0) {
      return res.status(400).json({
        error: '"statement" must not be empty.',
        code: "EMPTY_STATEMENT",
      });
    }

    if (cleaned.length > MAX_STATEMENT_LENGTH) {
      return res.status(400).json({
        error: `"statement" must be at most ${MAX_STATEMENT_LENGTH} characters.`,
        code: "STATEMENT_TOO_LONG",
      });
    }

    // --- Forward to FastAPI ---
    let upstream;
    try {
      upstream = await fetchWithTimeout(`${FASTAPI_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ statement: cleaned }),
      });
    } catch (err) {
      const { status, error, code } = describeUpstreamError(err);
      console.error(`[predict] upstream connection failure (${code}):`, err.cause?.code || err.message);
      return res.status(status).json({ error, code });
    }

    // Parse upstream body (may not be JSON on a crash)
    let data = null;
    try {
      data = await upstream.json();
    } catch {
      data = null;
    }

    if (!upstream.ok) {
      const detail = extractDetail(data);
      console.error(`[predict] FastAPI returned HTTP ${upstream.status}:`, detail || "(no detail)");
      // 4xx from FastAPI (e.g. 422) = our payload was rejected; surface as 400.
      // 5xx from FastAPI = model/service failure; surface as 502.
      const status = upstream.status >= 400 && upstream.status < 500 ? 400 : 502;
      return res.status(status).json({
        error: detail || "The ML service failed to process the statement.",
        code: status === 400 ? "ML_SERVICE_REJECTED_INPUT" : "ML_SERVICE_FAILURE",
      });
    }

    // Sanity-check the response shape — never invent or patch values.
    if (
      !data ||
      typeof data.prediction !== "string" ||
      typeof data.probabilities !== "object" ||
      data.probabilities === null
    ) {
      console.error("[predict] Unexpected FastAPI response shape:", data);
      return res.status(502).json({
        error: "The ML service returned an unexpected response format.",
        code: "ML_SERVICE_BAD_RESPONSE",
      });
    }

    // Return exactly what the model produced.
    return res.status(200).json({
      statement: data.statement ?? cleaned,
      prediction: data.prediction,
      probabilities: data.probabilities,
      disclaimer: data.disclaimer ?? "Model estimates, not verified facts.",
    });
  } catch (err) {
    return next(err);
  }
});

// ---------------------------------------------------------------------------
// 404 + error handlers
// ---------------------------------------------------------------------------
app.use((req, res) => {
  res.status(404).json({ error: `Route not found: ${req.method} ${req.originalUrl}`, code: "NOT_FOUND" });
});

// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  if (err.type === "entity.parse.failed") {
    return res.status(400).json({ error: "Malformed JSON in request body.", code: "INVALID_JSON" });
  }
  if (err.type === "entity.too.large") {
    return res.status(413).json({ error: "Request body too large.", code: "PAYLOAD_TOO_LARGE" });
  }
  if (err.message && err.message.startsWith("Origin not allowed by CORS")) {
    return res.status(403).json({ error: err.message, code: "CORS_FORBIDDEN" });
  }
  console.error("[server] Unhandled error:", err);
  return res.status(500).json({ error: "Internal server error.", code: "INTERNAL_ERROR" });
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
app.listen(PORT, () => {
  console.log(`TruthLens backend running on http://localhost:${PORT}`);
  console.log(`Forwarding predictions to ${FASTAPI_URL}/predict`);
});