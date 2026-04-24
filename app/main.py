from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.cors import CORSMiddleware
from app.routers.v1.api import api_router
from app.core.config import settings

from app.utils.logger import get_logger
logger = get_logger(__name__, log_file="app_main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    from app.utils.model_loader import load_models
    app.state.models = load_models()
    yield

app = FastAPI(title=settings.app_name, openapi_url=settings.api_base_path + "/openapi.json"
              ,version = settings.api_version, docs_url=settings.api_base_path+"/docs", openapi_version="3.1.0"
              ,redoc_url=None,lifespan=lifespan
              )

app.add_middleware(GZipMiddleware, compresslevel=9, minimum_size=0)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


logger.debug("API ROUTER INCLUSION STARTED")
app.include_router(api_router)

@app.get(settings.api_base_path+"/", include_in_schema=False)
async def root():
    return f"Welcome to Product Sales Forecasting API.  {settings.api_base_path}/docs"

#uvicorn app.main:app --reload --env-file .env