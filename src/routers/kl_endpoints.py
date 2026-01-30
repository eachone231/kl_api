from fastapi import APIRouter, Depends, HTTPException
import aiomysql

from src.resources.mysql import async_get_db
from src.model.kl_models import (
    CabinetsResponse,
    CabinetResponse,
    LoginData,
    LoginRequest,
    LoginResponse,
    MenuItemsRequest,
    MenuItemsResponse,
    ProjectsResponse,
)
from src.services.kl_service import (
    authenticate_user_async,
    fetch_cabinet_by_project_uuid_async,
    fetch_cabinets_by_project_async,
    fetch_active_menu_items_async,
    fetch_active_projects_async,
    health_status,
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
