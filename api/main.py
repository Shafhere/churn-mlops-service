import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.auth import require_api_key
from api.schemas import BatchInput, BatchOutput, CustomerInput, PredictionOutput
from api.service import ROOT, ModelService
from monitoring.store import MonitorStore

logger = logging.getLogger("churnguard.api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("CHURNGUARD_API_KEY"):
        raise RuntimeError("API key not configured")

    service = ModelService.load()
    app.state.model_service = service

    # Separate monitoring data for each model version.
    monitor_dir = Path(
        os.getenv(
            "CHURNGUARD_MONITOR_DB_DIR",
            str(ROOT / "artifacts"),
        )
    )

    app.state.monitor = MonitorStore(
        monitor_dir / f"monitoring_v{service.version}.sqlite",
        reference=getattr(service, "reference", None),
    )

    yield

    del app.state.model_service
    del app.state.monitor


app = FastAPI(
    title="ChurnGuard Prediction API",
    version="1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    """Track request latency, status and prediction totals."""

    started = time.perf_counter()
    status = 500

    try:
        response = await call_next(request)
        # Responses may contain sensitive prediction details.
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        status = response.status_code
        return response

    finally:
        elapsed = (time.perf_counter() - started) * 1000

        summary = getattr(
            request.state, "prediction_stats", {}
        )

        monitor = getattr(
            request.app.state, "monitor", None
        )

        if monitor is not None:
            monitor.record(
                route=request.url.path,
                status=status,
                latency_ms=elapsed,
                predictions=summary.get("count", 0),
                churn_yes=summary.get("yes", 0),
                feature_counts=getattr(
                    request.state, "feature_counts", {}
                ),
            )

        # Structured operational log, with no customer values.
        logger.info(json.dumps({
            "path": request.url.path,
            "status": status,
            "latency_ms": round(elapsed, 2),
            "predictions": summary.get("count", 0),
        }))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
):
    details = [
        {
            "field": ".".join(map(str, error["loc"])),
            "message": error["msg"],
        }
        for error in exc.errors()
    ]

    return JSONResponse(
        status_code=422,
        content={"detail": details},
    )


def get_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_monitor(request: Request) -> MonitorStore:
    return request.app.state.monitor


ServiceDep = Annotated[ModelService, Depends(get_service)]
MonitorDep = Annotated[MonitorStore, Depends(get_monitor)]


@app.get("/health")
def health(request: Request):
    return {
        "status": "healthy",
        "model_loaded": hasattr(
            request.app.state, "model_service"
        ),
    }


@app.get(
    "/model/info",
    dependencies=[Depends(require_api_key)],
)
def model_info(
    service: ServiceDep,
):
    return {
        "model_name": service.model_name,
        "model_version": service.version,
        "threshold": service.threshold,
        "test_metrics": service.test_metrics,
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    dependencies=[Depends(require_api_key)],
)
def predict(
    customer: CustomerInput,
    request: Request,
    service: ServiceDep,
    monitor: MonitorDep,
):
    output = service.predict([customer])[0]

    if hasattr(service, "explain"):
        output.top_factors = service.explain(customer)

    request.state.prediction_stats = {
        "count": 1,
        "yes": int(output.prediction == "Yes"),
    }

    request.state.feature_counts = (
        monitor.feature_counts([customer])
    )

    return output


@app.post(
    "/predict/batch",
    response_model=BatchOutput,
    dependencies=[Depends(require_api_key)],
)
def predict_batch(
    batch: BatchInput,
    request: Request,
    service: ServiceDep,
    monitor: MonitorDep,
):
    outputs = service.predict(batch.records)

    request.state.prediction_stats = {
        "count": len(outputs),
        "yes": sum(
            output.prediction == "Yes"
            for output in outputs
        ),
    }

    request.state.feature_counts = (
        monitor.feature_counts(batch.records)
    )

    return BatchOutput(
        count=len(outputs),
        predictions=outputs,
    )


@app.get(
    "/monitoring/summary",
    dependencies=[Depends(require_api_key)],
)
def monitoring_summary(
    monitor: MonitorDep,
):
    return monitor.summary()


@app.get(
    "/monitoring/drift",
    dependencies=[Depends(require_api_key)],
)
def monitoring_drift(
    monitor: MonitorDep,
):
    return monitor.drift()