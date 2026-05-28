from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Union

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.orchestrator import MAFISOrchestrator
from app.schemas import (
    AnalyzeRequest,
    AgentResponse,
    OrchestratorResponse,
    HealthResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    if not s.groq_api_key or s.groq_api_key == "your-groq-api-key-here":
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    if not s.fred_api_key or s.fred_api_key == "your-fred-api-key-here":
        raise RuntimeError("FRED_API_KEY is not set in .env")
    app.state.orchestrator = MAFISOrchestrator()
    print(f"MAFIS started | model={s.model_name} | env={s.environment}")
    yield


app = FastAPI(
    title="MAFIS — Multi-Agent Financial Intelligence System",
    description="Three-agent ReAct system for Rates, FX, and Equities analysis with NeMo Guardrails.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    print(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "Validation error", "detail": exc.errors()})


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service liveness check",
)
async def health():
    s = get_settings()
    return HealthResponse(status="ok", model=s.model_name, environment=s.environment)


@app.post(
    "/analyze/rates",
    response_model=AgentResponse,
    tags=["Analysis"],
    summary="Rates and monetary policy analysis",
)
async def analyze_rates(request: AnalyzeRequest):
    result = await asyncio.to_thread(app.state.orchestrator.run_agent, "rates", request.query)
    return result


@app.post(
    "/analyze/fx",
    response_model=AgentResponse,
    tags=["Analysis"],
    summary="Foreign exchange rate analysis",
)
async def analyze_fx(request: AnalyzeRequest):
    result = await asyncio.to_thread(app.state.orchestrator.run_agent, "fx", request.query)
    return result


@app.post(
    "/analyze/equities",
    response_model=AgentResponse,
    tags=["Analysis"],
    summary="Equity price and technical analysis",
)
async def analyze_equities(request: AnalyzeRequest):
    result = await asyncio.to_thread(app.state.orchestrator.run_agent, "equities", request.query)
    return result


@app.post(
    "/analyze",
    response_model=Union[AgentResponse, OrchestratorResponse],
    tags=["Analysis"],
    summary="Run all three agents concurrently",
)
async def analyze(request: AnalyzeRequest):
    if request.stream == "all":
        result = await app.state.orchestrator.run_all(request.query)
    else:
        result = await asyncio.to_thread(app.state.orchestrator.run_agent, request.stream, request.query)
    return result
