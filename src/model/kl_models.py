# schemas/menu.py
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict


class MenuItems(BaseModel):
    id: str          # menu_id
    label_k: str     # menu_k_name
    label_e: str     # menu_e_name
    order: int       # menu_order
    group: str | None = None       # optional grouping if present later


class MenuItemsResponse(BaseModel):
    items: list[MenuItems]


class MenuItemsRequest(BaseModel):
    emp_id: str


class Project(BaseModel):
    id: int
    name: str
    owner: str | None = None
    memo: str | None = None
    is_active: bool | None = None
    cabinet_count: int | None = None
    document_count: int | None = None
    updated_at: datetime | None = None
    created_at: datetime | None = None


class ProjectsResponse(BaseModel):
    items: list[Project]


class ProjectResponse(BaseModel):
    item: Project


class ProjectCreateRequest(BaseModel):
    project_name: str
    project_owner: str
    project_memo: str
    is_active: bool


class ProjectUpdateRequest(BaseModel):
    id: int
    project_name: str
    project_owner: str
    project_memo: str
    is_active: bool


class ProjectDeleteRequest(BaseModel):
    id: int


class ProjectDeleteResponse(BaseModel):
    deleted: bool


class ProjectsSummaryResponse(BaseModel):
    summary: dict[str, int]
    items: list[Project]


class ModelItem(BaseModel):
    id: int
    model_type: str
    provider: str
    model_name: str
    model_version: str | None = None
    dimension: int | None = None
    is_deprecated: bool
    created_at: datetime | None = None


class ModelsResponse(BaseModel):
    items: list[ModelItem]


class ModelsSummaryResponse(BaseModel):
    models_count: int
    llm_model_configs_count: int
    embedding_model_configs_count: int
    embedding_runs_count: int


class ModelUpdateRequest(BaseModel):
    id: int
    is_deprecated: bool


class ModelConfigRequest(BaseModel):
    id: int | None = None
    type: str | None = None


class ModelConfigResponse(BaseModel):
    model_id: int | None = None
    model_type: str | None = None
    configs: list[dict[str, object]] | None = None
    llm_configs: list[dict[str, object]] | None = None
    embedding_configs: list[dict[str, object]] | None = None


class LLMModelConfigCreateRequest(BaseModel):
    model_id: int
    temperature: float
    top_p: float
    max_tokens: int
    system_prompt: str


class LLMModelConfigResponse(BaseModel):
    config: dict[str, object]


class EmbeddingModelConfigCreateRequest(BaseModel):
    model_id: int
    chunk_size: float
    overlap: int
    normalize: int
    distance_metric: str
    length_unit: str


class EmbeddingModelConfigResponse(BaseModel):
    config: dict[str, object]


class LLMModelConfigDeleteRequest(BaseModel):
    id: int


class LLMModelConfigDeleteResponse(BaseModel):
    deleted: bool


class EmbeddingModelConfigDeleteRequest(BaseModel):
    id: int


class EmbeddingModelConfigDeleteResponse(BaseModel):
    deleted: bool


class StoragesResponse(BaseModel):
    items: list[str]


class VectorStoresResponse(BaseModel):
    items: list[str]


class SystemSecretViewRequest(BaseModel):
    secret_name: str
    provider: str
    env: str
    usage_scope: str


class SystemSecretViewResponse(BaseModel):
    secret_value: str


class SystemSecretCreateRequest(BaseModel):
    provider: str
    secret_name: str
    secret_value: str
    usage_scope: str
    env: str


class SystemSecretRotateRequest(BaseModel):
    secret_value: str


class SystemSecretDeleteRequest(BaseModel):
    id: int


class SystemSecretDeleteResponse(BaseModel):
    deleted: bool


class SystemSecretListItem(BaseModel):
    id: int
    provider: str
    secret_name: str
    secret_value: str
    usage_scope: str
    env: str
    is_active: bool
    rotated_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SystemSecretsResponse(BaseModel):
    items: list[SystemSecretListItem]
    summary: dict[str, int]


class SystemEnvironmentsResponse(BaseModel):
    items: list[str]


class SystemSecretProvidersResponse(BaseModel):
    items: list[str]


class Cabinet(BaseModel):
    id: int
    project_id: int
    cabinet_uuid: str
    name: str
    storage_type: str
    storage_base_path: str
    storage_path: str | None = None
    vector_store: str | None = None
    collection_name: str | None = None
    embedding_model_name: str | None = None
    embedding_dim: int | None = None
    is_active: bool | None = None


class CabinetsResponse(BaseModel):
    items: list[Cabinet]


class CabinetResponse(BaseModel):
    item: Cabinet


class CabinetUpdateRequest(BaseModel):
    project_id: int
    cabinet_uuid: str
    name: str
    storage_type: str
    storage_base_path: str
    storage_path: str
    vector_store: str
    collection_name: str
    embedding_model_name: str
    embedding_dim: int
    is_active: bool


class CabinetCreateRequest(BaseModel):
    project_id: int
    name: str
    storage_type: str
    storage_base_path: str
    storage_path: str
    vector_store: str
    collection_name: str
    embedding_model_id: int
    embedding_dim: int


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginData(BaseModel):
    emp_id: str
    username: str
    email: str


class LoginResponse(BaseModel):
    code: int
    data: LoginData | None = None
    message: str | None = None


class PageInfo(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int


class DocumentListItem(BaseModel):
    document_uuid: str
    file_name: str
    file_type: str
    file_size: int
    status: str
    processing_step: str | None = None
    uploaded_at: datetime | None = None


class DocumentsResponse(BaseModel):
    items: list[DocumentListItem]
    page_info: PageInfo


class DocumentSummaryResponse(BaseModel):
    total: int
    by_status: dict[str, int]


class UploadDocumentsResponse(BaseModel):
    items: list[DocumentListItem]


class DocumentDeleteRequest(BaseModel):
    document_uuid: str


class DocumentDeleteResponse(BaseModel):
    deleted: bool


class ChunkListItem(BaseModel):
    id: int
    doc_uuid: str
    document_name: str
    document_file_type: str | None = None
    chunking_run_id: int
    chunk_index: int | None = None
    content_preview: str | None = None
    content: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    chunking_config_id: int | None = None
    method_name: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    unit: str | None = None
    splitter_version: str | None = None


class ChunksResponse(BaseModel):
    items: list[ChunkListItem]
    page_info: PageInfo


class ChunkingConfig(BaseModel):
    id: int
    method_name: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    unit: str | None = None
    splitter_version: str | None = None
    memo: str | None = None


class ChunkingRun(BaseModel):
    id: int
    chunking_config_id: int
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    unit: str | None = None
    splitter_version: str | None = None
    memo: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChunkingRunCreate(BaseModel):
    chunking_config_id: int
    chunk_size: int
    chunk_overlap: int
    unit: str


class CabinetChunkingRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cabinet_uuid: str
    chunking_run: ChunkingRunCreate = Field(..., alias="chinking_run")


class CabinetChunkingRunResponse(BaseModel):
    cabinet_uuid: str
    chunking_run: ChunkingRun


class CabinetChunkingSettingsResponse(BaseModel):
    cabinet_uuid: str
    current_run: ChunkingRun | None = None
    configs: list[ChunkingConfig]


class CabinetChunkStatsResponse(BaseModel):
    cabinet_uuid: str
    total_chunks: int
    avg_chunk_length: float


class CabinetQASummaryResponse(BaseModel):
    cabinet_uuid: str
    total_documents: int
    total_qa: int
    evaluated_qa: int
    avg_score: float
    unevaluated_qa: int


class QAListItem(BaseModel):
    id: int
    chunk_id: int
    chunk_index: int | None = None
    document_uuid: str
    document_name: str
    question: str | None = None
    answer: str | None = None
    generated_by: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    evaluation_count: int
    avg_score: float | None = None
    evaluations: list["QAEvaluationItem"]


class QAListResponse(BaseModel):
    items: list[QAListItem]
    page_info: PageInfo


class QAEvaluationItem(BaseModel):
    evaluator_type: str | None = None
    score: int | None = None
    feedback: str | None = None
    evaluated_at: datetime | None = None


class QAEvaluationCreateRequest(BaseModel):
    qa_id: int
    evaluator_type: str
    score: int
    feedback: str


class QAEvaluationCreateResponse(BaseModel):
    evaluation_id: int | None = None
    qa_id: int
    evaluator_type: str
    score: int
    feedback: str


class DocumentQASummaryItem(BaseModel):
    file_name: str
    document_uuid: str
    created_at: datetime | None = None
    file_size: int | None = None
    chunks: int
    qa: int
    unevaluated_qa: int


class CabinetDocumentSummaryResponse(BaseModel):
    summary: dict[str, int]
    documents: list[DocumentQASummaryItem]
