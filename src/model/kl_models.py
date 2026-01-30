# schemas/menu.py
from pydantic import BaseModel


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


class ProjectsResponse(BaseModel):
    items: list[Project]


class Cabinet(BaseModel):
    id: int
    project_id: int
    uuid: str
    name: str
    storage_type: str
    storage_root_path: str
    storage_path: str | None = None
    vector_store: str | None = None
    collection_name: str | None = None
    embedding_model_name: str | None = None
    embedding_dim: int | None = None


class CabinetsResponse(BaseModel):
    items: list[Cabinet]


class CabinetResponse(BaseModel):
    item: Cabinet


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
