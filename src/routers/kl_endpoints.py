from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
import aiomysql

from src.resources.mysql import async_get_db
from src.model.kl_models import (
    CabinetsResponse,
    CabinetResponse,
    CabinetChunkingRunRequest,
    CabinetChunkingRunResponse,
    CabinetChunkingSettingsResponse,
    DocumentSummaryResponse,
    DocumentsResponse,
    LoginData,
    LoginRequest,
    LoginResponse,
    MenuItemsRequest,
    MenuItemsResponse,
    ProjectsResponse,
    UploadDocumentsResponse,
)
from src.services.kl_service import (
    authenticate_user_async,
    fetch_cabinet_by_uuid_async,
    fetch_cabinet_by_project_uuid_async,
    fetch_cabinets_by_project_async,
    fetch_active_menu_items_async,
    fetch_active_projects_async,
    health_status,
    fetch_documents_async,
    fetch_documents_summary_async,
    create_cabinet_chunking_run_async,
    fetch_cabinet_chunking_settings_async,
    save_uploaded_documents_async,
    say_hello,
)

api_router = APIRouter()


@api_router.get("/")
async def root():
    return {"message": "Hello World"}


@api_router.get("/hello/{name}")
async def hello(name: str):
    return {"message": say_hello(name)}


@api_router.get("/health")
async def health():
    return health_status()


@api_router.post("/api/login", tags=["auth"], response_model=LoginResponse)
async def login(
    payload: LoginRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    user = await authenticate_user_async(
        db,
        email=payload.email,
        password=payload.password,
    )
    if user is None:
        return LoginResponse(
            code=401,
            message="Invalid credentials",
        )
    return LoginResponse(
        code=0,
        data=LoginData(
            emp_id=user["emp_id"],
            username=user["username"],
            email=user["email"],
        ),
        message="OK",
    )


@api_router.post("/api/menu_items", tags=["menus"], response_model=MenuItemsResponse)
async def get_menu_items(
    payload: MenuItemsRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    items = await fetch_active_menu_items_async(db, emp_id=payload.emp_id)
    return MenuItemsResponse(items=items)


@api_router.get("/api/projects", tags=["projects"], response_model=ProjectsResponse)
async def get_projects(
    db: aiomysql.Connection = Depends(async_get_db),
):
    items = await fetch_active_projects_async(db)
    return ProjectsResponse(items=items)


@api_router.get("/api/cabinets", tags=["cabinets"], response_model=CabinetsResponse)
async def get_cabinets(
    project_id: int,
    db: aiomysql.Connection = Depends(async_get_db),
):
    items = await fetch_cabinets_by_project_async(db, project_id=project_id)
    return CabinetsResponse(items=items)


@api_router.get("/api/cabinet", tags=["cabinets"], response_model=CabinetResponse)
async def get_cabinet(
    project_id: int,
    cabinet_uuid: str,
    db: aiomysql.Connection = Depends(async_get_db),
):
    item = await fetch_cabinet_by_project_uuid_async(
        db,
        project_id=project_id,
        cabinet_uuid=cabinet_uuid,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Cabinet not found")
    return CabinetResponse(item=item)


@api_router.get(
    "/api/documents",
    tags=["documents"],
    response_model=DocumentsResponse,
)
async def get_documents(
    cabinet_uuid: str | None = Query(
        None,
        min_length=1,
        description="Filter documents by cabinet UUID; omit to return all documents.",
    ),
    page: int = Query(
        1,
        ge=1,
        description="Page number for pagination (1-based).",
    ),
    page_size: int = Query(
        5,
        ge=1,
        le=100,
        description="Page size for pagination (max 100).",
    ),
    status: str | None = Query(
        None,
        description="Filter documents by status.",
    ),
    file_type: str | None = Query(
        None,
        description="Filter documents by file type.",
    ),
    q: str | None = Query(
        None,
        description="Search documents by file name (substring match).",
    ),
    sort: str | None = Query(
        None,
        description=(
            "Sort option: created_at_desc, created_at_asc, file_name_asc, "
            "file_name_desc, size_desc, size_asc, status_asc, status_desc."
        ),
    ),
    db: aiomysql.Connection = Depends(async_get_db),
):
    items, total_items = await fetch_documents_async(
        db,
        cabinet_uuid=cabinet_uuid,
        page=page,
        page_size=page_size,
        status=status,
        file_type=file_type,
        query=q,
        sort=sort,
    )
    total_pages = (
        (total_items + page_size - 1) // page_size if total_items else 0
    )
    return DocumentsResponse(
        items=items,
        page_info={
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
        },
    )


@api_router.get(
    "/api/documents/summary",
    tags=["documents"],
    response_model=DocumentSummaryResponse,
)
async def get_documents_summary(
    cabinet_uuid: str = Query(..., min_length=1),
    db: aiomysql.Connection = Depends(async_get_db),
):
    total, by_status = await fetch_documents_summary_async(
        db,
        cabinet_uuid=cabinet_uuid,
    )
    return DocumentSummaryResponse(total=total, by_status=by_status)


@api_router.post(
    "/api/documents/upload",
    tags=["documents"],
    response_model=UploadDocumentsResponse,
)
async def upload_documents(
    cabinet_uuid: str = Form(..., min_length=1),
    files: list[UploadFile] = File(...),
    db: aiomysql.Connection = Depends(async_get_db),
):
    if not files or all(not (f.filename or "").strip() for f in files):
        raise HTTPException(status_code=400, detail="No files uploaded")
    cabinet = await fetch_cabinet_by_uuid_async(db, cabinet_uuid=cabinet_uuid)
    if cabinet is None:
        raise HTTPException(status_code=404, detail="Cabinet not found")
    items = await save_uploaded_documents_async(db, cabinet=cabinet, files=files)
    return UploadDocumentsResponse(items=items)


@api_router.post(
    "/api/cabinet/chunking",
    tags=["chunking"],
    response_model=CabinetChunkingRunResponse,
)
async def create_cabinet_chunking_run(
    payload: CabinetChunkingRunRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, chunking_run = await create_cabinet_chunking_run_async(
        db,
        cabinet_uuid=payload.cabinet_uuid,
        chunking_run=payload.chunking_run,
    )
    if not exists:
        raise HTTPException(status_code=404, detail="Cabinet not found")
    if chunking_run is None:
        raise HTTPException(
            status_code=500, detail="Failed to create chunking run"
        )
    return CabinetChunkingRunResponse(
        cabinet_uuid=payload.cabinet_uuid,
        chunking_run=chunking_run,
    )


@api_router.get(
    "/api/cabinet/chunking",
    tags=["chunking"],
    response_model=CabinetChunkingSettingsResponse,
)
async def get_cabinet_chunking_settings(
    cabinet_uuid: str = Query(..., min_length=1),
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, current_run, configs = (
        await fetch_cabinet_chunking_settings_async(
            db,
            cabinet_uuid=cabinet_uuid,
        )
    )
    if not exists:
        raise HTTPException(status_code=404, detail="Cabinet not found")
    return CabinetChunkingSettingsResponse(
        cabinet_uuid=cabinet_uuid,
        current_run=current_run,
        configs=configs,
    )
