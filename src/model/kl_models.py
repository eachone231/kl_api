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


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginData(BaseModel):
    emp_id: str
    email: str


class LoginResponse(BaseModel):
    code: int
    data: LoginData | None = None
    message: str | None = None
