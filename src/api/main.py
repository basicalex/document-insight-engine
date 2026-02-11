from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from src.config.settings import Settings, settings
from src.ingestion import UploadIntakeError, UploadIntakeService
from src.models.schemas import IngestResponse


app = FastAPI(title=settings.project_name)


def get_app_settings() -> Settings:
    return settings


@app.on_event("startup")
def ensure_runtime_dirs() -> None:
    settings.ensure_runtime_dirs()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(
    file: UploadFile = File(...),
    cfg: Settings = Depends(get_app_settings),
) -> IngestResponse:
    intake = UploadIntakeService(cfg)
    try:
        receipt = await intake.save_upload(file)
    except UploadIntakeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return IngestResponse(
        document_id=receipt.document_id, file_path=str(receipt.file_path)
    )
