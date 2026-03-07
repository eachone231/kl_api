from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    Request,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
from datetime import datetime, timezone
import aiomysql
import json
import tempfile

from src.resources.mysql import async_get_db
from src.model.kl_models import (
    CabinetsResponse,
    CabinetResponse,
    CabinetDeleteRequest,
    CabinetDeleteResponse,
    CabinetChunkingRunRequest,
    CabinetChunkingRunResponse,
    CabinetChunkingSettingsResponse,
    CabinetChunkStatsResponse,
    CabinetQASummaryResponse,
    CabinetDocumentSummaryResponse,
    CabinetCreateRequest,
    CabinetUpdateRequest,
    QAListResponse,
    QAEvaluationCreateRequest,
    QAEvaluationCreateResponse,
    DocumentQAGenerationRequest,
    DocumentQAGenerationResponse,
    DocumentSummaryResponse,
    DocumentsResponse,
    ChunksResponse,
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    LoginData,
    LoginRequest,
    LoginResponse,
    ModelsResponse,
    ModelsSummaryResponse,
    SystemProfilesResponse,
    SystemProfileCreateRequest,
    SystemProfileResponse,
    SystemProfileDeleteRequest,
    SystemProfileDeleteResponse,
    SystemProfileUpdateRequest,
    ModelUpdateRequest,
    ModelConfigRequest,
    ModelConfigResponse,
    LLMModelConfigCreateRequest,
    LLMModelConfigResponse,
    EmbeddingModelConfigCreateRequest,
    EmbeddingModelConfigResponse,
    LLMModelConfigDeleteRequest,
    LLMModelConfigDeleteResponse,
    EmbeddingModelConfigDeleteRequest,
    EmbeddingModelConfigDeleteResponse,
    ProjectsSummaryResponse,
    StoragesResponse,
    VectorStoresResponse,
    MenuItemsRequest,
    MenuItemsResponse,
    ProjectsResponse,
    ProjectCreateRequest,
    ProjectResponse,
    ProjectUpdateRequest,
    ProjectDeleteRequest,
    ProjectDeleteResponse,
    SystemSecretCreateRequest,
    SystemSecretDeleteRequest,
    SystemSecretDeleteResponse,
    SystemSecretListItem,
    SystemSecretRotateRequest,
    SystemSecretViewRequest,
    SystemSecretViewResponse,
    SystemSecretsResponse,
    SystemEnvironmentsResponse,
    SystemSecretProvidersResponse,
    SystemVdbPatchRequest,
    UploadDocumentsResponse,
    UserCreateRequest,
    UserResponse,
    UserResetRequest,
    UserResetResponse,
    UserResetConfirmRequest,
    UserResetConfirmResponse,
    EnquerySummaryResponse,
    EnqueryListResponse,
)
from src.services.kl_service import (
    authenticate_user_async,
    fetch_cabinet_by_uuid_async,
    fetch_cabinet_by_project_uuid_async,
    fetch_cabinets_by_project_async,
    fetch_active_menu_items_async,
    fetch_active_projects_async,
    fetch_models_async,
    fetch_models_summary_async,
    fetch_system_profiles_async,
    create_system_profile_async,
    delete_system_profile_async,
    update_system_profile_async,
    deactivate_system_profile_async,
    activate_system_profile_async,
    update_model_deprecated_async,
    fetch_model_configs_async,
    create_llm_model_config_async,
    create_embedding_model_config_async,
    update_llm_model_config_async,
    update_embedding_model_config_async,
    delete_llm_model_config_async,
    delete_embedding_model_config_async,
    fetch_projects_summary_async,
    create_project_async,
    update_project_async,
    delete_project_async,
    fetch_storage_types_async,
    fetch_vector_stores_async,
    fetch_system_secret_async,
    fetch_system_secrets_async,
    fetch_system_environments_async,
    fetch_system_vdb_profiles_async,
    fetch_system_secret_providers_async,
    create_system_secret_async,
    rotate_system_secret_async,
    delete_system_secret_async,
    upsert_system_vdb_profile_async,
    health_status,
    fetch_documents_async,
    fetch_document_download_info_async,
    fetch_documents_summary_async,
    delete_document_async,
    create_cabinet_chunking_run_async,
    fetch_cabinet_chunking_settings_async,
    fetch_cabinet_chunk_stats_async,
    fetch_cabinet_qa_summary_async,
    fetch_cabinet_qa_list_async,
    create_qa_evaluation_async,
    create_cabinet_async,
    update_cabinet_async,
    activate_cabinet_async,
    deactivate_cabinet_async,
    delete_cabinet_async,
    fetch_cabinet_document_summary_async,
    fetch_chunks_async,
    save_uploaded_documents_async,
    say_hello,
    create_user_async,
    create_user_reset_token_async,
    confirm_user_reset_token_async,
    fetch_enquery_summary_async,
    fetch_enqueries_async,
    enqueue_document_qa_generation_async,
    enqueue_document_pipeline_async,
    enqueue_cabinet_collection_delete_async,
)
from src.resources.crypto_env import encrypt_secret
from src.resources.auth import create_access_token
from src.resources.redis import build_redis_client, get_redis_client
from src.config import settings

api_router = APIRouter()
_RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"


def _load_resource_json(filename: str):
    path = _RESOURCE_DIR / filename
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Resource not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON resource")


def _write_resource_json(filename: str, data):
    path = _RESOURCE_DIR / filename
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(path)
    except OSError:
        raise HTTPException(status_code=500, detail="Failed to write resource")


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
    user, reason = await authenticate_user_async(
        db,
        email=payload.email,
        password=payload.password,
    )
    if user is None:
        if reason == "inactive":
            return LoginResponse(
                code=403,
                message="승인 대기중",
            )
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
        access_token=create_access_token(
            emp_id=user["emp_id"],
            email=user["email"],
        ),
        token_type="bearer",
        message="OK",
    )


@api_router.post("/api/user", tags=["users"], response_model=UserResponse)
async def create_user(
    payload: UserCreateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    ok, user, reason = await create_user_async(db, payload=payload)
    if not ok:
        if reason == "emp_id":
            raise HTTPException(
                status_code=409,
                detail={"message": "이미 등록된 사원번호"},
            )
        if reason == "email":
            raise HTTPException(
                status_code=409,
                detail={"message": "이미 등록된 이메일"},
            )
        if reason in {"emp_id,email", "email,emp_id"}:
            raise HTTPException(
                status_code=409,
                detail={"message": "이미 등록된 사용자"},
            )
        raise HTTPException(
            status_code=500,
            detail={"message": reason or "unknown"},
        )
    return UserResponse(item=user)


@api_router.post(
    "/api/user/reset",
    tags=["users"],
    response_model=UserResetResponse,
)
async def create_user_reset(
    payload: UserResetRequest,
    request: Request,
    db: aiomysql.Connection = Depends(async_get_db),
):
    ip_address = request.client.host if request.client else None
    ok, data, reason = await create_user_reset_token_async(
        db, payload=payload, ip_address=ip_address
    )
    if not ok:
        if reason == "user_not_found":
            raise HTTPException(
                status_code=404, detail={"message": "User not found"}
            )
        raise HTTPException(
            status_code=500, detail={"message": reason or "unknown"}
        )
    return UserResetResponse(
        token=data["token"],
        expires_at=data["expires_at"],
    )


@api_router.post(
    "/api/user/reset/confirm",
    tags=["users"],
    response_model=UserResetConfirmResponse,
)
async def confirm_user_reset(
    payload: UserResetConfirmRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    ok, reason = await confirm_user_reset_token_async(db, payload=payload)
    if not ok:
        if reason in {"invalid_token", "user_not_found"}:
            raise HTTPException(
                status_code=404, detail={"message": "Invalid token"}
            )
        if reason == "token_expired":
            raise HTTPException(
                status_code=410, detail={"message": "Token expired"}
            )
        if reason == "token_used":
            raise HTTPException(
                status_code=409, detail={"message": "Token already used"}
            )
        raise HTTPException(
            status_code=500, detail={"message": reason or "unknown"}
        )
    return UserResetConfirmResponse(reset=True)


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


@api_router.post("/api/project", tags=["projects"], response_model=ProjectResponse)
async def create_project(
    payload: ProjectCreateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    try:
        project = await create_project_async(db, payload=payload)
    except aiomysql.IntegrityError:
        raise HTTPException(status_code=409, detail="Project already exists")
    return ProjectResponse(item=project)


@api_router.delete(
    "/api/project",
    tags=["projects"],
    response_model=ProjectDeleteResponse,
)
async def delete_project(
    payload: ProjectDeleteRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    deleted, reason = await delete_project_async(db, project_id=payload.id)
    if not deleted:
        if reason == "not_found":
            raise HTTPException(status_code=404, detail="Project not found")
        if reason == "has_cabinets":
            raise HTTPException(
                status_code=409,
                detail="Project has cabinets",
            )
        if reason == "has_documents":
            raise HTTPException(
                status_code=409,
                detail="Project has documents",
            )
        raise HTTPException(status_code=500, detail="Failed to delete project")
    return ProjectDeleteResponse(deleted=True)


@api_router.put("/api/project", tags=["projects"], response_model=ProjectResponse)
async def update_project(
    payload: ProjectUpdateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, project = await update_project_async(db, payload=payload)
    if not exists:
        raise HTTPException(status_code=404, detail="Project not found")
    if project is None:
        raise HTTPException(status_code=500, detail="Failed to update project")
    return ProjectResponse(item=project)


@api_router.get(
    "/api/projects/summary",
    tags=["projects"],
    response_model=ProjectsSummaryResponse,
)
async def get_projects_summary(
    db: aiomysql.Connection = Depends(async_get_db),
):
    summary, items = await fetch_projects_summary_async(db)
    return ProjectsSummaryResponse(summary=summary, items=items)


@api_router.get("/api/models", tags=["models"], response_model=ModelsResponse)
async def get_models(
    type: str | None = Query(
        None,
        description="Filter by model type: llm or embedding.",
    ),
    status: str | None = Query(
        None,
        description="Filter by model status: active.",
    ),
    db: aiomysql.Connection = Depends(async_get_db),
):
    model_type = None
    if type:
        normalized = type.strip().lower()
        if normalized == "llm":
            model_type = "LLM"
        elif normalized == "embedding":
            model_type = "EMBEDDING"
        else:
            raise HTTPException(
                status_code=400,
                detail="type must be 'llm' or 'embedding'",
            )
    status_filter = None
    if status:
        normalized_status = status.strip().lower()
        if normalized_status == "active":
            status_filter = "active"
        else:
            raise HTTPException(
                status_code=400, detail="status must be 'active'"
            )
    items = await fetch_models_async(
        db, model_type=model_type, status_filter=status_filter
    )
    return ModelsResponse(items=items)


@api_router.get(
    "/api/models/summary",
    tags=["models"],
    response_model=ModelsSummaryResponse,
)
async def get_models_summary(
    db: aiomysql.Connection = Depends(async_get_db),
):
    summary = await fetch_models_summary_async(db)
    return ModelsSummaryResponse(**summary)


@api_router.get(
    "/api/enquery/summary",
    tags=["enquery"],
    response_model=EnquerySummaryResponse,
)
async def get_enquery_summary(
    db: aiomysql.Connection = Depends(async_get_db),
):
    summary = await fetch_enquery_summary_async(db)
    return EnquerySummaryResponse(**summary)


@api_router.get(
    "/api/enqueries",
    tags=["enquery"],
    response_model=EnqueryListResponse,
)
async def get_enqueries(
    page: int = Query(
        1,
        ge=1,
        description="Page number for pagination (1-based).",
    ),
    page_size: int = Query(
        10,
        ge=1,
        le=100,
        description="Page size for pagination (max 100).",
    ),
    db: aiomysql.Connection = Depends(async_get_db),
):
    items, total_items = await fetch_enqueries_async(
        db,
        page=page,
        page_size=page_size,
    )
    total_pages = (
        (total_items + page_size - 1) // page_size if total_items else 0
    )
    return EnqueryListResponse(
        items=items,
        page_info={
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
        },
    )


@api_router.get(
    "/api/model/system_profile",
    tags=["models"],
    response_model=SystemProfilesResponse,
)
async def get_system_profiles(
    is_active: bool | None = Query(
        None,
        description="Filter profiles by active status.",
    ),
    db: aiomysql.Connection = Depends(async_get_db),
):
    items = await fetch_system_profiles_async(db, is_active=is_active)
    return SystemProfilesResponse(items=items)


@api_router.post(
    "/api/model/system_profile",
    tags=["models"],
    response_model=SystemProfileResponse,
)
async def create_system_profile(
    payload: SystemProfileCreateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    try:
        item = await create_system_profile_async(db, payload=payload)
    except aiomysql.IntegrityError:
        raise HTTPException(
            status_code=409, detail="System profile already exists"
        )
    return SystemProfileResponse(item=item)


@api_router.delete(
    "/api/model/system_profile",
    tags=["models"],
    response_model=SystemProfileDeleteResponse,
)
async def delete_system_profile(
    payload: SystemProfileDeleteRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    deleted = await delete_system_profile_async(
        db, system_profile_id=payload.id
    )
    if not deleted:
        raise HTTPException(
            status_code=404, detail="System profile not found"
        )
    return SystemProfileDeleteResponse(deleted=True)


@api_router.put(
    "/api/model/system_profile",
    tags=["models"],
    response_model=SystemProfileResponse,
)
async def update_system_profile(
    payload: SystemProfileUpdateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, item = await update_system_profile_async(db, payload=payload)
    if not exists:
        raise HTTPException(
            status_code=404, detail={"message": "System profile not found"}
        )
    if item is None:
        raise HTTPException(
            status_code=500,
            detail={"message": "Failed to update system profile"},
        )
    return SystemProfileResponse(item=item)


@api_router.patch(
    "/api/model/system_profile/{system_profile_id}/deactivate",
    tags=["models"],
    response_model=SystemProfileResponse,
)
async def deactivate_system_profile(
    system_profile_id: int,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, item, cabinets, reason = await deactivate_system_profile_async(
        db, system_profile_id=system_profile_id
    )
    if not exists:
        raise HTTPException(status_code=404, detail="System profile not found")
    if reason == "has_active_cabinets":
        raise HTTPException(
            status_code=409,
            detail={
                "message": "System profile has active cabinets",
                "cabinets": jsonable_encoder(cabinets),
            },
        )
    if item is None:
        raise HTTPException(
            status_code=500, detail="Failed to update system profile"
        )
    return SystemProfileResponse(item=item)


@api_router.patch(
    "/api/model/system_profile/{system_profile_id}/activate",
    tags=["models"],
    response_model=SystemProfileResponse,
)
async def activate_system_profile(
    system_profile_id: int,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, item, reason = await activate_system_profile_async(
        db, system_profile_id=system_profile_id
    )
    if not exists:
        raise HTTPException(
            status_code=404, detail={"message": "System profile not found"}
        )
    if item is None:
        raise HTTPException(
            status_code=500,
            detail={"message": "Failed to update system profile"},
        )
    return SystemProfileResponse(item=item)


@api_router.patch("/api/model", tags=["models"], response_model=ModelsResponse)
async def update_model_deprecated(
    payload: ModelUpdateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, item = await update_model_deprecated_async(db, payload=payload)
    if not exists:
        raise HTTPException(status_code=404, detail="Model not found")
    if item is None:
        raise HTTPException(status_code=500, detail="Failed to update model")
    return ModelsResponse(items=[item])


@api_router.get(
    "/api/model/configs",
    tags=["models"],
    response_model=ModelConfigResponse,
)
async def get_model_config(
    id: int | None = Query(None),
    model_type: str | None = Query(None),
    db: aiomysql.Connection = Depends(async_get_db),
):
    if id is None and model_type is None:
        raise HTTPException(
            status_code=400, detail="id or model_type must be provided"
        )
    exists, model_type, configs, configs_by_type = (
        await fetch_model_configs_async(
            db, model_id=id, model_type=model_type
        )
    )
    if id is not None and not exists:
        raise HTTPException(status_code=404, detail="Model not found")
    if model_type is None:
        raise HTTPException(status_code=500, detail="Model type missing")
    if configs_by_type is not None:
        return ModelConfigResponse(
            model_type=model_type,
            llm_configs=configs_by_type.get("llm_configs"),
            embedding_configs=configs_by_type.get("embedding_configs"),
        )
    if configs is None:
        raise HTTPException(status_code=404, detail="Model config not found")
    return ModelConfigResponse(
        model_id=id, model_type=model_type, configs=configs
    )


@api_router.post(
    "/api/model/config/llm",
    tags=["models"],
    response_model=LLMModelConfigResponse,
)
async def create_llm_model_config(
    payload: LLMModelConfigCreateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    try:
        config = await create_llm_model_config_async(db, payload=payload)
    except aiomysql.IntegrityError:
        raise HTTPException(
            status_code=409, detail="Model config already exists"
        )
    return LLMModelConfigResponse(config=config)


@api_router.put(
    "/api/model/config/llm",
    tags=["models"],
    response_model=LLMModelConfigResponse,
)
async def update_llm_model_config(
    payload: LLMModelConfigCreateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, config = await update_llm_model_config_async(db, payload=payload)
    if not exists:
        raise HTTPException(status_code=404, detail="Model config not found")
    if config is None:
        raise HTTPException(
            status_code=500, detail="Failed to update model config"
        )
    return LLMModelConfigResponse(config=config)


@api_router.post(
    "/api/model/config/embedding",
    tags=["models"],
    response_model=EmbeddingModelConfigResponse,
)
async def create_embedding_model_config(
    payload: EmbeddingModelConfigCreateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    try:
        config = await create_embedding_model_config_async(
            db, payload=payload
        )
    except aiomysql.IntegrityError:
        raise HTTPException(
            status_code=409, detail="Model config already exists"
        )
    return EmbeddingModelConfigResponse(config=config)


@api_router.put(
    "/api/model/config/embedding",
    tags=["models"],
    response_model=EmbeddingModelConfigResponse,
)
async def update_embedding_model_config(
    payload: EmbeddingModelConfigCreateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, config = await update_embedding_model_config_async(
        db, payload=payload
    )
    if not exists:
        raise HTTPException(status_code=404, detail="Model config not found")
    if config is None:
        raise HTTPException(
            status_code=500, detail="Failed to update model config"
        )
    return EmbeddingModelConfigResponse(config=config)


@api_router.delete(
    "/api/model/config/llm",
    tags=["models"],
    response_model=LLMModelConfigDeleteResponse,
)
async def delete_llm_model_config(
    payload: LLMModelConfigDeleteRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    deleted = await delete_llm_model_config_async(db, config_id=payload.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Model config not found")
    return LLMModelConfigDeleteResponse(deleted=True)


@api_router.delete(
    "/api/model/config/embedding",
    tags=["models"],
    response_model=EmbeddingModelConfigDeleteResponse,
)
async def delete_embedding_model_config(
    payload: EmbeddingModelConfigDeleteRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    deleted = await delete_embedding_model_config_async(
        db, config_id=payload.id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Model config not found")
    return EmbeddingModelConfigDeleteResponse(deleted=True)


@api_router.get("/api/storages", tags=["storages"], response_model=StoragesResponse)
async def get_storages(
    db: aiomysql.Connection = Depends(async_get_db),
):
    items = await fetch_storage_types_async(db)
    return StoragesResponse(items=items)


@api_router.get(
    "/api/vector_stores",
    tags=["vector_stores"],
    response_model=VectorStoresResponse,
)
async def get_vector_stores(
    db: aiomysql.Connection = Depends(async_get_db),
):
    items = await fetch_vector_stores_async(db)
    return VectorStoresResponse(items=items)


@api_router.get("/api/system/vdb", tags=["system"])
async def get_system_vdb(
    db: aiomysql.Connection = Depends(async_get_db),
):
    envs = await fetch_system_environments_async(db)
    profiles = await fetch_system_vdb_profiles_async(db)
    response = {
        "vector_stores": [
            "qdrant",
            "chromadb",
            "weaviate",
            "milvus",
            "faiss",
            "pinecone",
        ]
    }
    for env in envs:
        profile = profiles.get(env, {})
        response[env] = {
            "vector_store": profile.get("vector_store", ""),
            "host": profile.get("host", ""),
            "port": profile.get("port", ""),
            "token": profile.get("token", ""),
            "updated_at": profile.get("updated_at", ""),
        }
    return response


@api_router.patch("/api/system/vdb", tags=["system"])
async def patch_system_vdb(
    payload: SystemVdbPatchRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    envs = await fetch_system_environments_async(db)
    env = payload.env
    if env not in envs:
        raise HTTPException(status_code=400, detail="Invalid env")
    existing_profiles = await fetch_system_vdb_profiles_async(db)
    existing = existing_profiles.get(env, {})
    token = payload.token
    if token is None:
        token = existing.get("token", "")
    elif token == "":
        token = ""
    elif token.startswith("ENC"):
        token = existing.get("token", "")
    profile = {
        "vector_store": payload.vector_store,
        "host": payload.host,
        "port": payload.port,
        "token": encrypt_secret(token) if token else "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await upsert_system_vdb_profile_async(db, env, profile)
    return profile


@api_router.post(
    "/api/system/secret/view",
    tags=["system_secrets"],
    response_model=SystemSecretViewResponse,
)
async def view_system_secret(
    payload: SystemSecretViewRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    secret_value = await fetch_system_secret_async(
        db,
        key=payload.secret_name,
        provider=payload.provider,
        env=payload.env,
        usage_scope=payload.usage_scope,
    )
    if secret_value is None:
        raise HTTPException(status_code=404, detail="Secret not found")
    return SystemSecretViewResponse(secret_value=secret_value)


@api_router.post(
    "/api/system/secrets",
    tags=["system_secrets"],
    response_model=SystemSecretListItem,
)
async def create_system_secret(
    payload: SystemSecretCreateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    try:
        item = await create_system_secret_async(db, payload=payload)
    except aiomysql.IntegrityError:
        raise HTTPException(status_code=409, detail="Secret already exists")
    return item


@api_router.patch(
    "/api/system/secrets/{secret_id}",
    tags=["system_secrets"],
    response_model=SystemSecretListItem,
)
async def rotate_system_secret(
    secret_id: int,
    payload: SystemSecretRotateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    item = await rotate_system_secret_async(
        db, secret_id=secret_id, secret_value=payload.secret_value
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Secret not found")
    return item


@api_router.delete(
    "/api/system/secrets",
    tags=["system_secrets"],
    response_model=SystemSecretDeleteResponse,
)
async def delete_system_secret(
    payload: SystemSecretDeleteRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    deleted = await delete_system_secret_async(db, secret_id=payload.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Secret not found")
    return SystemSecretDeleteResponse(deleted=True)


@api_router.get(
    "/api/system/secrets",
    tags=["system_secrets"],
    response_model=SystemSecretsResponse,
)
async def list_system_secrets(
    env: str | None = Query(
        None,
        description="Filter secrets by environment.",
    ),
    db: aiomysql.Connection = Depends(async_get_db),
):
    items, stats = await fetch_system_secrets_async(db, env=env)
    return SystemSecretsResponse(items=items, summary=stats)


@api_router.get(
    "/api/system/environments",
    tags=["system_environments"],
    response_model=SystemEnvironmentsResponse,
)
async def list_system_environments(
    db: aiomysql.Connection = Depends(async_get_db),
):
    items = await fetch_system_environments_async(db)
    return SystemEnvironmentsResponse(items=items)


@api_router.get(
    "/api/system/secret_providers",
    tags=["system_secrets"],
    response_model=SystemSecretProvidersResponse,
)
async def list_system_secret_providers(
    db: aiomysql.Connection = Depends(async_get_db),
):
    items = await fetch_system_secret_providers_async(db)
    return SystemSecretProvidersResponse(items=items)


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


@api_router.post("/api/cabinet", tags=["cabinets"], response_model=CabinetResponse)
async def create_cabinet(
    payload: CabinetCreateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    ok, cabinet = await create_cabinet_async(db, payload=payload)
    if not ok:
        raise HTTPException(status_code=404, detail="System profile not found")
    if cabinet is None:
        raise HTTPException(status_code=500, detail="Failed to create cabinet")
    return CabinetResponse(item=cabinet)


@api_router.put("/api/cabinet", tags=["cabinets"], response_model=CabinetResponse)
async def update_cabinet(
    payload: CabinetUpdateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, cabinet, reason = await update_cabinet_async(db, payload=payload)
    if not exists:
        if reason == "system_profile_missing":
            raise HTTPException(
                status_code=404, detail="System profile not found"
            )
        raise HTTPException(status_code=404, detail="Cabinet not found")
    if cabinet is None:
        raise HTTPException(
            status_code=500, detail="Failed to update cabinet"
        )
    return CabinetResponse(item=cabinet)


@api_router.delete(
    "/api/cabinet",
    tags=["cabinets"],
    response_model=CabinetDeleteResponse,
)
async def delete_cabinet(
    payload: CabinetDeleteRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    import logging

    logger = logging.getLogger(__name__)
    cabinet = await fetch_cabinet_by_uuid_async(
        db, cabinet_uuid=payload.cabinet_uuid
    )
    deleted, reason = await delete_cabinet_async(
        db, cabinet_uuid=payload.cabinet_uuid
    )
    if not deleted:
        if reason == "not_found":
            raise HTTPException(status_code=404, detail="Cabinet not found")
        if reason == "has_documents":
            raise HTTPException(
                status_code=409, detail="Cabinet has documents"
            )
        raise HTTPException(status_code=500, detail="Failed to delete cabinet")
    await db.commit()

    if cabinet is not None:
        try:
            redis_client = await get_redis_client()
            await enqueue_cabinet_collection_delete_async(
                redis_client=redis_client,
                cabinet_uuid=cabinet.cabinet_uuid,
                vector_store=cabinet.vector_store,
                collection_name=cabinet.collection_name,
            )
        except RuntimeError as exc:
            logger.error(
                "cabinet delete enqueue failed: cabinet_uuid=%s runtime_error=%s",
                payload.cabinet_uuid,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail="Cabinet deleted but failed to enqueue collection deletion",
            )
        except Exception as exc:
            logger.exception(
                "cabinet delete enqueue failed: cabinet_uuid=%s unexpected_error=%s",
                payload.cabinet_uuid,
                exc,
            )
            raise HTTPException(
                status_code=500,
                detail="Cabinet deleted but failed to enqueue collection deletion",
            )
    return CabinetDeleteResponse(deleted=True)


@api_router.patch(
    "/api/cabinet/{cabinet_uuid}/activate",
    tags=["cabinets"],
    response_model=CabinetResponse,
)
async def activate_cabinet(
    cabinet_uuid: str,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, cabinet, system_profile, reason = await activate_cabinet_async(
        db, cabinet_uuid=cabinet_uuid
    )
    if not exists:
        raise HTTPException(
            status_code=404, detail={"message": "Cabinet not found"}
        )
    if reason == "system_profile_missing":
        raise HTTPException(
            status_code=404, detail={"message": "System profile not found"}
        )
    if reason == "system_profile_inactive":
        raise HTTPException(
            status_code=409,
            detail={
                "message": "System profile is inactive",
                "system_profile": jsonable_encoder(system_profile),
            },
        )
    if cabinet is None:
        raise HTTPException(
            status_code=500, detail={"message": "Failed to update cabinet"}
        )
    return CabinetResponse(item=cabinet)


@api_router.patch(
    "/api/cabinet/{cabinet_uuid}/deactivate",
    tags=["cabinets"],
    response_model=CabinetResponse,
)
async def deactivate_cabinet(
    cabinet_uuid: str,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, cabinet, reason = await deactivate_cabinet_async(
        db, cabinet_uuid=cabinet_uuid
    )
    if not exists:
        raise HTTPException(
            status_code=404, detail={"message": "Cabinet not found"}
        )
    if cabinet is None:
        raise HTTPException(
            status_code=500, detail={"message": "Failed to update cabinet"}
        )
    return CabinetResponse(item=cabinet)


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


@api_router.get("/api/documents/download", tags=["documents"])
async def download_document(
    document_uuid: str = Query(..., min_length=1),
    db: aiomysql.Connection = Depends(async_get_db),
):
    info = await fetch_document_download_info_async(
        db, document_uuid=document_uuid
    )
    if info is None:
        raise HTTPException(status_code=404, detail="Document not found")
    file_path = info["file_path"]
    file_name = info["file_name"]
    try:
        return FileResponse(
            path=file_path,
            filename=file_name,
            media_type="application/octet-stream",
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")


@api_router.delete(
    "/api/document",
    tags=["documents"],
    response_model=DocumentDeleteResponse,
)
async def delete_document(
    payload: DocumentDeleteRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    deleted, reason = await delete_document_async(
        db, document_uuid=payload.document_uuid
    )
    if not deleted:
        if reason == "not_found":
            raise HTTPException(status_code=404, detail="Document not found")
        if reason == "parsing":
            raise HTTPException(
                status_code=409,
                detail="문서 변환중에는 삭제할 수 없읍니다",
            )
        if reason == "file_delete_failed":
            raise HTTPException(
                status_code=500, detail="Failed to delete document file"
            )
        raise HTTPException(status_code=500, detail="Failed to delete document")
    return DocumentDeleteResponse(deleted=True)


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


@api_router.get(
    "/api/cabinet/chunks",
    tags=["chunks"],
    response_model=ChunksResponse,
)
async def get_chunks(
    cabinet_uuid: str | None = Query(
        None,
        min_length=1,
        description="Filter chunks by cabinet UUID; omit to return all chunks.",
    ),
    page: int = Query(
        1,
        ge=1,
        description="Page number for pagination (1-based).",
    ),
    page_size: int = Query(
        10,
        ge=1,
        le=100,
        description="Page size for pagination (max 100).",
    ),
    preview_length: int = Query(
        120,
        ge=20,
        le=1000,
        description="Preview length for content snippet.",
    ),
    document_uuid: str | None = Query(
        None,
        description="Filter chunks by document UUID.",
    ),
    chunking_run_id: int | None = Query(
        None,
        description="Filter chunks by chunking run id.",
    ),
    sort: str | None = Query(
        None,
        description=(
            "Sort option: created_at_desc, created_at_asc, index_asc, "
            "index_desc, doc_name_asc, doc_name_desc, chunk_id_asc, "
            "chunk_id_desc."
        ),
    ),
    db: aiomysql.Connection = Depends(async_get_db),
):
    items, total_items = await fetch_chunks_async(
        db,
        cabinet_uuid=cabinet_uuid,
        page=page,
        page_size=page_size,
        preview_length=preview_length,
        document_uuid=document_uuid,
        chunking_run_id=chunking_run_id,
        sort=sort,
    )
    total_pages = (
        (total_items + page_size - 1) // page_size if total_items else 0
    )
    return ChunksResponse(
        items=items,
        page_info={
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
        },
    )


@api_router.get(
    "/api/cabinet/chunks/stats",
    tags=["chunks"],
    response_model=CabinetChunkStatsResponse,
)
async def get_cabinet_chunk_stats(
    cabinet_uuid: str = Query(..., min_length=1),
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, total_chunks, avg_chunk_length = (
        await fetch_cabinet_chunk_stats_async(
            db,
            cabinet_uuid=cabinet_uuid,
        )
    )
    if not exists:
        raise HTTPException(status_code=404, detail="Cabinet not found")
    return CabinetChunkStatsResponse(
        cabinet_uuid=cabinet_uuid,
        total_chunks=total_chunks,
        avg_chunk_length=avg_chunk_length,
    )


@api_router.get(
    "/api/cabinet/qa/summary",
    tags=["qa"],
    response_model=CabinetQASummaryResponse,
)
async def get_cabinet_qa_summary(
    cabinet_uuid: str = Query(..., min_length=1),
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, total_documents, total_qa, evaluated_qa, avg_score, unevaluated_qa = (
        await fetch_cabinet_qa_summary_async(
            db,
            cabinet_uuid=cabinet_uuid,
        )
    )
    if not exists:
        raise HTTPException(status_code=404, detail="Cabinet not found")
    return CabinetQASummaryResponse(
        cabinet_uuid=cabinet_uuid,
        total_documents=total_documents,
        total_qa=total_qa,
        evaluated_qa=evaluated_qa,
        avg_score=avg_score,
        unevaluated_qa=unevaluated_qa,
    )


@api_router.get(
    "/api/cabinet/qa",
    tags=["qa"],
    response_model=QAListResponse,
)
async def get_cabinet_qa_list(
    cabinet_uuid: str = Query(..., min_length=1),
    document_uuid: str | None = Query(None, min_length=1),
    page: int = Query(
        1,
        ge=1,
        description="Page number for pagination (1-based).",
    ),
    page_size: int = Query(
        10,
        ge=1,
        le=100,
        description="Page size for pagination (max 100).",
    ),
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, items, total_items = await fetch_cabinet_qa_list_async(
        db,
        cabinet_uuid=cabinet_uuid,
        document_uuid=document_uuid,
        page=page,
        page_size=page_size,
    )
    if not exists:
        raise HTTPException(status_code=404, detail="Cabinet not found")
    total_pages = (
        (total_items + page_size - 1) // page_size if total_items else 0
    )
    return QAListResponse(
        items=items,
        page_info={
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
        },
    )


@api_router.post(
    "/api/cabinet/qa/evaluate",
    tags=["qa"],
    response_model=QAEvaluationCreateResponse,
)
async def create_qa_evaluation(
    payload: QAEvaluationCreateRequest,
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, evaluation_id = await create_qa_evaluation_async(
        db,
        qa_id=payload.qa_id,
        evaluator_type=payload.evaluator_type,
        score=payload.score,
        feedback=payload.feedback,
    )
    if not exists:
        raise HTTPException(status_code=404, detail="QA not found")
    return QAEvaluationCreateResponse(
        evaluation_id=evaluation_id,
        qa_id=payload.qa_id,
        evaluator_type=payload.evaluator_type,
        score=payload.score,
        feedback=payload.feedback,
    )


@api_router.get(
    "/api/cabinet/document/summary",
    tags=["documents"],
    response_model=CabinetDocumentSummaryResponse,
)
async def get_cabinet_document_summary(
    cabinet_uuid: str = Query(..., min_length=1),
    db: aiomysql.Connection = Depends(async_get_db),
):
    exists, total_documents, total_qa, documents = (
        await fetch_cabinet_document_summary_async(
            db,
            cabinet_uuid=cabinet_uuid,
        )
    )
    if not exists:
        raise HTTPException(status_code=404, detail="Cabinet not found")
    return CabinetDocumentSummaryResponse(
        summary={
            "total_documents": total_documents,
            "total_qa": total_qa,
        },
        documents=documents,
    )


@api_router.post(
    "/api/document/qa-gen",
    tags=["documents"],
    response_model=DocumentQAGenerationResponse,
)
async def enqueue_document_qa_generation(
    payload: DocumentQAGenerationRequest,
):
    stream = settings.redis_stream_key or settings.redis_stream
    if not stream:
        return DocumentQAGenerationResponse(
            enqueued=False,
            doc_uuid=payload.doc_uuid,
            error="REDIS_STREAM_KEY is not configured",
        )
    try:
        redis_client = await get_redis_client()
        entry_id = await enqueue_document_qa_generation_async(
            redis_client=redis_client,
            doc_uuid=payload.doc_uuid,
            stream=stream,
        )
    except Exception as exc:
        return DocumentQAGenerationResponse(
            enqueued=False,
            doc_uuid=payload.doc_uuid,
            error=str(exc),
        )
    return DocumentQAGenerationResponse(
        enqueued=True,
        doc_uuid=payload.doc_uuid,
        stream_entry_id=entry_id,
    )


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
    import logging
    import os
    from pathlib import Path

    logger = logging.getLogger(__name__)

    def _should_expose_error_detail() -> bool:
        return settings.expose_internal_error_detail or settings.env.lower() in {
            "local",
            "dev",
            "development",
        }

    def _error_detail(prefix: str, exc: Exception) -> str:
        if _should_expose_error_detail():
            return f"{prefix}: {exc.__class__.__name__}: {exc}"
        return prefix

    if not files or all(not (f.filename or "").strip() for f in files):
        raise HTTPException(status_code=400, detail="No files uploaded")
    for file in files:
        filename = (file.filename or "").strip().lower()
        content_type = (file.content_type or "").strip().lower()
        is_pdf_name = filename.endswith(".pdf")
        is_pdf_type = content_type in ("application/pdf", "application/x-pdf")
        if not (is_pdf_name or is_pdf_type):
            raise HTTPException(
                status_code=400,
                detail="PDF 형식의 파일만 지원합니다. PDF 형식으로 변환하여 등록해주십시요.",
            )
    cabinet = await fetch_cabinet_by_uuid_async(db, cabinet_uuid=cabinet_uuid)
    if cabinet is None:
        raise HTTPException(status_code=404, detail="Cabinet not found")
    if cabinet.is_active is False:
        raise HTTPException(status_code=403, detail="Cabinet is inactive")
    if not cabinet.storage_base_path:
        raise HTTPException(
            status_code=400, detail="Cabinet storage_base_path is missing"
        )
    if not cabinet.storage_path:
        raise HTTPException(
            status_code=400, detail="Cabinet storage_path is missing"
        )
    if not cabinet.vector_store:
        raise HTTPException(
            status_code=400, detail="Cabinet vector_store is missing"
        )
    if cabinet.system_profile is None:
        raise HTTPException(
            status_code=400, detail="Cabinet system_profile is missing"
        )

    base_dir = Path(cabinet.storage_base_path)
    rel_path = Path(cabinet.storage_path)
    if rel_path.is_absolute():
        raise HTTPException(
            status_code=400, detail="Cabinet storage_path must be relative"
        )
    target_dir = base_dir / rel_path
    if ".." in target_dir.parts:
        raise HTTPException(
            status_code=400, detail="Cabinet storage_path is invalid"
        )

    logger.info(
        "upload validation paths: cabinet_uuid=%s base=%r storage_path=%r target=%r base_exists=%s base_is_dir=%s target_exists=%s target_parent=%r",
        cabinet.cabinet_uuid,
        cabinet.storage_base_path,
        cabinet.storage_path,
        str(target_dir),
        base_dir.exists(),
        base_dir.is_dir(),
        target_dir.exists(),
        str(target_dir.parent),
    )

    def _is_writable_dir(path: Path) -> bool:
        if path.exists():
            return path.is_dir() and os.access(path, os.W_OK)
        parent = path.parent
        return parent.exists() and os.access(parent, os.W_OK)

    if base_dir.is_absolute() and not _is_writable_dir(base_dir):
        logger.warning(
            "upload validation failed: cabinet_uuid=%s base=%r exists=%s is_dir=%s writable=%s parent=%r parent_exists=%s parent_writable=%s",
            cabinet.cabinet_uuid,
            cabinet.storage_base_path,
            base_dir.exists(),
            base_dir.is_dir(),
            os.access(base_dir, os.W_OK),
            str(base_dir.parent),
            base_dir.parent.exists(),
            os.access(base_dir.parent, os.W_OK),
        )
        raise HTTPException(
            status_code=403,
            detail=(
                "Cabinet storage_base_path is not writable or accessible"
            ),
        )
    if target_dir.is_absolute() and not _is_writable_dir(target_dir):
        logger.warning(
            "upload validation failed: cabinet_uuid=%s target=%r exists=%s is_dir=%s writable=%s parent=%r parent_exists=%s parent_writable=%s",
            cabinet.cabinet_uuid,
            str(target_dir),
            target_dir.exists(),
            target_dir.is_dir(),
            os.access(target_dir, os.W_OK),
            str(target_dir.parent),
            target_dir.parent.exists(),
            os.access(target_dir.parent, os.W_OK),
        )
        raise HTTPException(
            status_code=403,
            detail="Cabinet storage_path is not writable or accessible",
        )
    try:
        items = await save_uploaded_documents_async(
            db, cabinet=cabinet, files=files
        )
        # Commit before enqueue so workers can read the document row.
        await db.commit()
    except RuntimeError as exc:
        logger.warning(
            "upload failed: cabinet_uuid=%s error=%s",
            cabinet.cabinet_uuid,
            exc,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Upload failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception(
            "upload failed: cabinet_uuid=%s unexpected_error=%s",
            cabinet.cabinet_uuid,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Upload failed: unexpected error", exc),
        ) from exc
    try:
        redis_client = await get_redis_client()
        await enqueue_document_pipeline_async(
            redis_client,
            cabinet_uuid=cabinet.cabinet_uuid,
            items=items,
        )
    except RuntimeError as exc:
        logger.exception(
            "enqueue failed: cabinet_uuid=%s runtime_error=%s",
            cabinet.cabinet_uuid,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Failed to enqueue documents", exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "enqueue failed: cabinet_uuid=%s unexpected_error=%s",
            cabinet.cabinet_uuid,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=_error_detail(
                "Failed to enqueue documents: unexpected error",
                exc,
            ),
        ) from exc
    return UploadDocumentsResponse(items=items)


@api_router.get(
    "/api/pipeline/status",
    tags=["pipeline"],
)
async def pipeline_status_sse(request: Request):
    stream = settings.redis_pipeline_status_stream
    if not stream:
        raise HTTPException(
            status_code=500,
            detail="REDIS_PIPELINE_STATUS_STREAM is not configured",
        )
    try:
        redis_client = build_redis_client()
        await redis_client.connect()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Redis is not configured: {exc}",
        ) from exc

    async def event_stream():
        last_id = "$"
        keepalive_every_ms = 30000
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    result = await redis_client.xread(
                        stream=stream,
                        last_id=last_id,
                        block_ms=keepalive_every_ms,
                        count=100,
                    )
                except Exception as exc:
                    payload = {
                        "error": str(exc),
                        "stream": stream,
                    }
                    yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                    break

                if not result:
                    yield ": keep-alive\n\n"
                    continue

                for stream_name, entries in result:
                    for entry_id, fields in entries:
                        data = {}
                        if isinstance(fields, list):
                            it = iter(fields)
                            for key in it:
                                try:
                                    value = next(it)
                                except StopIteration:
                                    value = ""
                                data[str(key)] = value
                        payload = {
                            "stream": stream_name,
                            "id": entry_id,
                            "fields": data,
                        }
                        last_id = entry_id
                        yield f"data: {json.dumps(payload)}\n\n"
        finally:
            await redis_client.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


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
