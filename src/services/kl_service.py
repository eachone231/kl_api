def say_hello(name: str) -> str:
    return f"Hello {name}"


def health_status() -> dict:
    return {"status": "ok"}


async def fetch_active_menu_items_async(db, emp_id: str) -> list["MenuItems"]:
    import aiomysql

    from src.model.kl_models import MenuItems

    qry = """
    SELECT DISTINCT
        m.menu_id, m.menu_k_name, m.menu_e_name, m.menu_order
    FROM users u 
        JOIN user_groups_map ugm ON ugm.user_id = u.id
        JOIN user_group_permissions ugp ON ugp.group_id = ugm.group_id
        JOIN menu_item_permissions mip ON mip.id = ugp.permission_id
        JOIN menu_item_permissions_map mipm ON mipm.permission_id = mip.id
        JOIN menu_items m ON m.id = mipm.menu_item_id
    WHERE
        u.emp_id = %(emp_id)s AND m.is_active = 1
    ORDER BY m.menu_order
    LIMIT 0 , 1000
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, {"emp_id": emp_id})
        rows = await cursor.fetchall()
    return [
        MenuItems(
            id=row["menu_id"],
            label_k=row["menu_k_name"],
            label_e=row["menu_e_name"],
            order=row["menu_order"],
        )
        for row in rows
    ]


async def fetch_active_projects_async(db) -> list["Project"]:
    import aiomysql

    from src.model.kl_models import Project

    qry = """
    SELECT
        id, project_name, project_owner, project_memo
    FROM projects
    WHERE is_active = 1
    ORDER BY id
    LIMIT 0 , 1000
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry)
        rows = await cursor.fetchall()
    return [
        Project(
            id=row["id"],
            name=row["project_name"],
            owner=row["project_owner"],
            memo=row["project_memo"],
        )
        for row in rows
    ]


async def fetch_cabinets_by_project_async(db, project_id: int) -> list["Cabinet"]:
    import aiomysql

    from src.model.kl_models import Cabinet

    storage_cols = await _resolve_cabinet_storage_columns_async(db)
    storage_type_col = storage_cols["storage_type"]
    base_path_col = storage_cols["storage_base_path"]
    upload_path_col = storage_cols["storage_path"]
    qry = """
    SELECT
        id,
        project_id,
        cabinet_uuid,
        cabinet_name,
        {storage_type_col} AS storage_type,
        {base_path_col} AS storage_base_path,
        {upload_path_col} AS storage_path,
        vector_store,
        collection_name,
        embedding_model_name,
        embedding_dim
    FROM cabinets
    WHERE project_id = %(project_id)s
    ORDER BY id
    LIMIT 0 , 1000
    """.format(
        storage_type_col=storage_type_col,
        base_path_col=base_path_col,
        upload_path_col=upload_path_col,
    )
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, {"project_id": project_id})
        rows = await cursor.fetchall()
    return [
        Cabinet(
            id=row["id"],
            project_id=row["project_id"],
            uuid=row["cabinet_uuid"],
            cabinet_uuid=row["cabinet_uuid"],
            name=row["cabinet_name"],
            storage_type=row["storage_type"],
            storage_root_path=row["storage_base_path"],
            storage_path=row["storage_path"],
            vector_store=row["vector_store"],
            collection_name=row["collection_name"],
            embedding_model_name=row["embedding_model_name"],
            embedding_dim=row["embedding_dim"],
        )
        for row in rows
    ]


async def fetch_cabinet_by_project_uuid_async(
    db,
    project_id: int,
    cabinet_uuid: str,
) -> "Cabinet | None":
    import aiomysql

    from src.model.kl_models import Cabinet

    storage_cols = await _resolve_cabinet_storage_columns_async(db)
    storage_type_col = storage_cols["storage_type"]
    base_path_col = storage_cols["storage_base_path"]
    upload_path_col = storage_cols["storage_path"]
    qry = """
    SELECT
        id,
        project_id,
        cabinet_uuid,
        cabinet_name,
        {storage_type_col} AS storage_type,
        {base_path_col} AS storage_base_path,
        {upload_path_col} AS storage_path,
        vector_store,
        collection_name,
        embedding_model_name,
        embedding_dim
    FROM cabinets
    WHERE project_id = %(project_id)s
        AND cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """.format(
        storage_type_col=storage_type_col,
        base_path_col=base_path_col,
        upload_path_col=upload_path_col,
    )
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(
            qry,
            {"project_id": project_id, "cabinet_uuid": cabinet_uuid},
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    return Cabinet(
        id=row["id"],
        project_id=row["project_id"],
        uuid=row["cabinet_uuid"],
        cabinet_uuid=row["cabinet_uuid"],
        name=row["cabinet_name"],
        storage_type=row["storage_type"],
        storage_root_path=row["storage_base_path"],
        storage_path=row["storage_path"],
        vector_store=row["vector_store"],
        collection_name=row["collection_name"],
        embedding_model_name=row["embedding_model_name"],
        embedding_dim=row["embedding_dim"],
    )


async def fetch_cabinet_by_uuid_async(
    db,
    cabinet_uuid: str,
) -> "Cabinet | None":
    import aiomysql

    from src.model.kl_models import Cabinet

    storage_cols = await _resolve_cabinet_storage_columns_async(db)
    storage_type_col = storage_cols["storage_type"]
    base_path_col = storage_cols["storage_base_path"]
    upload_path_col = storage_cols["storage_path"]
    qry = """
    SELECT
        id,
        project_id,
        cabinet_uuid,
        cabinet_name,
        {storage_type_col} AS storage_type,
        {base_path_col} AS storage_base_path,
        {upload_path_col} AS storage_path,
        vector_store,
        collection_name,
        embedding_model_name,
        embedding_dim
    FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """.format(
        storage_type_col=storage_type_col,
        base_path_col=base_path_col,
        upload_path_col=upload_path_col,
    )
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, {"cabinet_uuid": cabinet_uuid})
        row = await cursor.fetchone()
    if row is None:
        return None
    return Cabinet(
        id=row["id"],
        project_id=row["project_id"],
        uuid=row["cabinet_uuid"],
        cabinet_uuid=row["cabinet_uuid"],
        name=row["cabinet_name"],
        storage_type=row["storage_type"],
        storage_root_path=row["storage_base_path"],
        storage_path=row["storage_path"],
        vector_store=row["vector_store"],
        collection_name=row["collection_name"],
        embedding_model_name=row["embedding_model_name"],
        embedding_dim=row["embedding_dim"],
    )


def hash_password(password: str) -> str:
    from passlib.hash import argon2

    return argon2.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    from passlib.hash import argon2

    return argon2.verify(password, password_hash)


async def authenticate_user_async(db, email: str, password: str) -> dict | None:
    import aiomysql

    qry = """
    SELECT id, emp_id, username, email, password_hash
    FROM users
    WHERE email = %(email)s AND is_active = 1
    LIMIT 1
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, {"email": email})
        row = await cursor.fetchone()
    if row is None:
        return None
    password_hash = row.get("password_hash")
    if not password_hash or not verify_password(password, password_hash):
        return None
    return row


def _normalize_documents_sort(sort: str | None) -> str:
    sort_map = {
        "createdAt_desc": "d.created_at DESC",
        "createdAt_asc": "d.created_at ASC",
        "file_name_asc": "d.file_name ASC",
        "file_name_desc": "d.file_name DESC",
        "size_desc": "d.file_size DESC",
        "size_asc": "d.file_size ASC",
        "status_asc": "d.status ASC",
        "status_desc": "d.status DESC",
    }
    return sort_map.get(sort or "", "d.created_at DESC")


def _pick_first_column(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


async def _resolve_cabinet_storage_columns_async(
    db,
) -> dict[str, str]:
    columns = await _get_cabinets_columns_async(db)
    column_names = set(columns.keys())
    storage_type = _pick_first_column(
        column_names,
        ("storage_type",),
    )
    base_path = _pick_first_column(
        column_names,
        ("storage_base_path", "storage_root_path", "base_path", "root_path"),
    )
    upload_path = _pick_first_column(
        column_names,
        ("storage_path", "storage_base", "upload_path", "storage_upload_path"),
    )
    if not storage_type or not base_path or not upload_path:
        missing = [
            name
            for name, value in (
                ("storage_type", storage_type),
                ("storage_base_path", base_path),
                ("storage_path", upload_path),
            )
            if value is None
        ]
        raise RuntimeError(
            "cabinets table is missing expected storage columns: "
            + ", ".join(missing)
        )
    return {
        "storage_type": f"`{storage_type}`",
        "storage_base_path": f"`{base_path}`",
        "storage_path": f"`{upload_path}`",
    }


def _first_enum_value(column_type: str | None) -> str | None:
    if not column_type or not column_type.startswith("enum("):
        return None
    values = column_type[5:-1]
    if not values:
        return None
    first = values.split(",", 1)[0].strip()
    if len(first) >= 2 and first[0] == "'" and first[-1] == "'":
        return first[1:-1]
    return None


async def _get_documents_columns_async(
    db,
) -> dict[str, dict[str, str | None]]:
    import aiomysql

    global _DOCUMENTS_COLUMNS
    if _DOCUMENTS_COLUMNS is not None:
        return _DOCUMENTS_COLUMNS

    qry = """
    SELECT
        COLUMN_NAME AS column_name,
        IS_NULLABLE AS is_nullable,
        COLUMN_DEFAULT AS column_default,
        EXTRA AS extra,
        COLUMN_TYPE AS column_type
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name = 'documents'
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry)
        rows = await cursor.fetchall()

    _DOCUMENTS_COLUMNS = {
        row["column_name"]: {
            "is_nullable": row["is_nullable"],
            "column_default": row["column_default"],
            "extra": row["extra"],
            "column_type": row["column_type"],
        }
        for row in rows
    }
    return _DOCUMENTS_COLUMNS


async def save_uploaded_documents_async(
    db,
    cabinet: "Cabinet",
    files: list["UploadFile"],
) -> list["DocumentListItem"]:
    import os
    import logging
    from datetime import datetime
    from pathlib import Path
    from uuid import uuid4

    import aiomysql

    from fastapi import UploadFile
    from src.model.kl_models import DocumentListItem

    logger = logging.getLogger(__name__)

    logger.info(
        "upload received files=%s types=%s",
        len(files) if files else 0,
        [type(f).__name__ for f in files] if files else [],
    )
    if not files:
        return []

    if not cabinet.storage_root_path:
        raise RuntimeError("Cabinet storage_root_path is missing")
    storage_type = (cabinet.storage_type or "").upper()
    if storage_type and storage_type not in ("LOCAL", "FILESYSTEM", "NFS"):
        raise RuntimeError(f"Unsupported storage_type: {cabinet.storage_type}")

    def _is_writable_dir(path: Path) -> bool:
        if path.exists():
            return path.is_dir() and os.access(path, os.W_OK)
        parent = path.parent
        return parent.exists() and os.access(parent, os.W_OK)

    base_dir = Path(cabinet.storage_root_path)
    if base_dir.is_absolute() and not _is_writable_dir(base_dir):
        fallback_dir = (Path.cwd() / "upload").resolve()
        if _is_writable_dir(fallback_dir):
            base_dir = fallback_dir
        else:
            raise RuntimeError(
                f"storage_root_path is not writable: {cabinet.storage_root_path}"
            )
    if cabinet.storage_path:
        storage_path = Path(cabinet.storage_path)
        if storage_path.is_absolute():
            storage_path = Path(str(storage_path).lstrip("/"))
        base_dir = base_dir / storage_path
    if base_dir.is_absolute() and not _is_writable_dir(base_dir):
        fallback_dir = (Path.cwd() / "upload").resolve()
        if _is_writable_dir(fallback_dir):
            base_dir = fallback_dir
        else:
            raise RuntimeError(
                f"storage_root_path is not writable: {base_dir}"
            )
    logger.info(
        "upload target resolved: root=%s storage_path=%s final=%s",
        cabinet.storage_root_path,
        cabinet.storage_path,
        base_dir,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    columns = await _get_documents_columns_async(db)
    cabinet_ref_column = None
    if "cabinet_uuid" in columns:
        cabinet_ref_column = "cabinet_uuid"
    elif "cabinet_id" in columns:
        cabinet_ref_column = "cabinet_id"
    else:
        raise RuntimeError("documents table is missing cabinet_uuid/cabinet_id")

    items: list[DocumentListItem] = []
    now = datetime.utcnow()

    async with db.cursor(aiomysql.DictCursor) as cursor:
        for upload in files:
            if not hasattr(upload, "filename") or not hasattr(upload, "read"):
                logger.warning(
                    "upload skipped: unsupported type=%s module=%s",
                    type(upload).__name__,
                    type(upload).__module__,
                )
                continue
            original_name = os.path.basename(upload.filename or "upload")
            if not original_name:
                original_name = "upload"
            suffix = Path(original_name).suffix
            doc_id = str(uuid4())
            saved_name = f"{doc_id}{suffix}"
            target_path = base_dir / saved_name
            if target_path.exists():
                doc_id = str(uuid4())
                saved_name = f"{doc_id}{suffix}"
                target_path = base_dir / saved_name

            size = 0
            try:
                await upload.seek(0)
            except Exception:
                pass
            with target_path.open("wb") as handle:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    size += len(chunk)

            await upload.close()
            logger.info(
                "upload saved: name=%s path=%s size=%s",
                saved_name,
                target_path,
                size,
            )

            suffix = Path(saved_name).suffix.lower()
            file_type = suffix.lstrip(".").lower() or "bin"
            values: dict[str, object] = {
                "id": doc_id,
                "file_name": original_name,
                "file_type": file_type,
                "file_size": size,
                "status": None,
                "processing_step": None,
                "created_at": now,
                "updated_at": now,
                "storage_path": str(target_path),
                "file_path": str(target_path),
                "path": str(target_path),
            }
            if cabinet_ref_column == "cabinet_uuid":
                values["cabinet_uuid"] = cabinet.cabinet_uuid
            else:
                values["cabinet_id"] = cabinet.id

            insert_columns = []
            params: dict[str, object] = {}
            for key, value in values.items():
                if key in columns and value is not None:
                    insert_columns.append(key)
                    params[key] = value

            required_missing = []
            for name, meta in columns.items():
                if (
                    meta["is_nullable"] == "NO"
                    and meta["column_default"] is None
                    and "auto_increment" not in (meta["extra"] or "")
                ):
                    if name not in params:
                        required_missing.append(name)

            for name in list(required_missing):
                meta = columns[name]
                enum_value = _first_enum_value(meta["column_type"])
                if name in ("status", "processing_step") and enum_value:
                    params[name] = enum_value
                    insert_columns.append(name)
                    required_missing.remove(name)
                elif name == "status":
                    params[name] = "UPLOADED"
                    insert_columns.append(name)
                    required_missing.remove(name)
                elif name == "processing_step":
                    params[name] = "UPLOADED"
                    insert_columns.append(name)
                    required_missing.remove(name)
                elif name in ("created_at", "updated_at"):
                    params[name] = now
                    insert_columns.append(name)
                    required_missing.remove(name)
                elif name == "id":
                    column_type = (meta.get("column_type") or "").lower()
                    int_types = (
                        "int",
                        "bigint",
                        "smallint",
                        "mediumint",
                        "tinyint",
                    )
                    if column_type.startswith(int_types):
                        raise RuntimeError(
                            "documents.id must be AUTO_INCREMENT or have a default"
                        )
                    params[name] = str(uuid4())
                    insert_columns.append(name)
                    required_missing.remove(name)

            if required_missing:
                missing = ", ".join(required_missing)
                raise RuntimeError(
                    f"documents table requires columns not provided: {missing}"
                )

            columns_sql = ", ".join(insert_columns)
            values_sql = ", ".join([f"%({col})s" for col in insert_columns])
            insert_qry = (
                f"INSERT INTO documents ({columns_sql}) VALUES ({values_sql})"
            )
            await cursor.execute(insert_qry, params)
            doc_id = cursor.lastrowid
            status_value = params.get("status") or (
                columns.get("status", {}).get("column_default") or "UPLOADED"
            )
            step_value = params.get("processing_step") or (
                columns.get("processing_step", {}).get("column_default") or "PENDING"
            )

            items.append(
                DocumentListItem(
                    id=str(doc_id),
                    file_name=original_name,
                    file_type=file_type,
                    file_size=size,
                    status=str(status_value),
                    processing_step=str(step_value),
                    uploaded_at=now,
                )
            )

    return items


async def fetch_documents_async(
    db,
    cabinet_uuid: str | None,
    page: int,
    page_size: int,
    status: str | None,
    file_type: str | None,
    query: str | None,
    sort: str | None,
) -> tuple[list["DocumentListItem"], int]:
    import aiomysql

    from src.model.kl_models import DocumentListItem

    join_sql = ""
    where_clauses = ["1=1"]
    params: dict[str, object] = {}
    if cabinet_uuid:
        join_sql, base_where_sql = await _resolve_documents_cabinet_filter_async(db)
        base_clause = base_where_sql.replace("WHERE ", "", 1)
        where_clauses = [base_clause]
        params["cabinet_uuid"] = cabinet_uuid
    if status:
        where_clauses.append("d.status = %(status)s")
        params["status"] = status
    if file_type:
        where_clauses.append("d.file_type = %(file_type)s")
        params["file_type"] = file_type
    if query:
        where_clauses.append("d.file_name LIKE %(query)s")
        params["query"] = f"%{query}%"

    where_sql = " AND ".join(where_clauses)
    sort_sql = _normalize_documents_sort(sort)
    offset = (page - 1) * page_size
    params.update({"offset": offset, "limit": page_size})

    count_qry = f"""
    SELECT COUNT(*) AS total
    FROM documents d
    {join_sql}
    WHERE {where_sql}
    """
    list_qry = f"""
    SELECT
        d.id,
        d.file_name,
        d.file_type,
        d.file_size,
        d.status,
        d.processing_step,
        d.created_at
    FROM documents d
    {join_sql}
    WHERE {where_sql}
    ORDER BY {sort_sql}
    LIMIT %(offset)s, %(limit)s
    """

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(count_qry, params)
        count_row = await cursor.fetchone()
        total_items = int(count_row["total"]) if count_row else 0
        await cursor.execute(list_qry, params)
        rows = await cursor.fetchall()

    items = [
        DocumentListItem(
            id=row["id"],
            file_name=row["file_name"],
            file_type=row["file_type"],
            file_size=row["file_size"],
            status=row["status"],
            processing_step=row["processing_step"],
            uploaded_at=row["created_at"],
        )
        for row in rows
    ]
    return items, total_items


async def fetch_documents_summary_async(
    db,
    cabinet_uuid: str,
) -> tuple[int, dict[str, int]]:
    import aiomysql

    join_sql, where_sql = await _resolve_documents_cabinet_filter_async(db)
    qry = f"""
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN d.processing_step = 'COMPLETE' THEN 1 ELSE 0 END) AS completed_count,
        SUM(
            CASE
                WHEN d.processing_step IN (
                    'CHUNKING',
                    'EMBEDDING',
                    'PARSING',
                    'QA_GENERATING'
                ) THEN 1
                ELSE 0
            END
        ) AS processing_count,
        SUM(CASE WHEN d.processing_step = 'FAIL' THEN 1 ELSE 0 END) AS failed_count
    FROM documents d
    {join_sql}
    {where_sql}
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, {"cabinet_uuid": cabinet_uuid})
        row = await cursor.fetchone()

    row = row or {}
    total = int(row.get("total") or 0)
    by_status = {
        "COMPLETED": int(row.get("completed_count") or 0),
        "PROCESSING": int(row.get("processing_count") or 0),
        "FAILED": int(row.get("failed_count") or 0),
    }
    return total, by_status


async def fetch_cabinet_chunking_settings_async(
    db,
    cabinet_uuid: str,
) -> tuple[bool, "ChunkingRun | None", list["ChunkingConfig"]]:
    import aiomysql

    from src.model.kl_models import ChunkingConfig, ChunkingRun

    cabinet_qry = """
    SELECT id
    FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    configs_qry = """
    SELECT
        id,
        method_name,
        chunk_size,
        chunk_overlap,
        unit,
        splitter_version,
        memo
    FROM chunking_configs
    ORDER BY id
    """
    current_run_qry = """
    SELECT
        id,
        chunking_config_id,
        cabinet_uuid,
        chunk_size,
        chunk_overlap,
        unit,
        splitter_version,
        memo,
        created_at,
        updated_at
    FROM chunking_runs
    WHERE cabinet_uuid = %(cabinet_uuid)s
    ORDER BY created_at DESC, id DESC
    LIMIT 1
    """

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(cabinet_qry, {"cabinet_uuid": cabinet_uuid})
        cabinet_row = await cursor.fetchone()
        if cabinet_row is None:
            return False, None, []

        await cursor.execute(configs_qry)
        configs_rows = await cursor.fetchall()

        current_run = None
        await cursor.execute(current_run_qry, {"cabinet_uuid": cabinet_uuid})
        row = await cursor.fetchone()
        if row:
            current_run = ChunkingRun(**row)

    configs = [ChunkingConfig(**row) for row in configs_rows]
    return True, current_run, configs


async def create_cabinet_chunking_run_async(
    db,
    cabinet_uuid: str,
    chunking_run: "ChunkingRunCreate",
) -> tuple[bool, "ChunkingRun | None"]:
    import aiomysql
    from datetime import datetime

    from src.model.kl_models import ChunkingRun

    cabinet_qry = """
    SELECT id
    FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    insert_qry = """
    INSERT INTO chunking_runs (
        chunking_config_id,
        cabinet_uuid,
        chunk_size,
        chunk_overlap,
        unit,
        splitter_version,
        memo
    ) VALUES (
        %(chunking_config_id)s,
        %(cabinet_uuid)s,
        %(chunk_size)s,
        %(chunk_overlap)s,
        %(unit)s,
        %(splitter_version)s,
        %(memo)s
    )
    """
    select_qry = """
    SELECT
        id,
        chunking_config_id,
        cabinet_uuid,
        chunk_size,
        chunk_overlap,
        unit,
        splitter_version,
        memo,
        created_at,
        updated_at
    FROM chunking_runs
    WHERE cabinet_uuid = %(cabinet_uuid)s
    ORDER BY created_at DESC, id DESC
    LIMIT 1
    """

    params = {
        "chunking_config_id": chunking_run.chunking_config_id,
        "cabinet_uuid": cabinet_uuid,
        "chunk_size": chunking_run.chunk_size,
        "chunk_overlap": chunking_run.chunk_overlap,
        "unit": chunking_run.unit,
        "splitter_version": chunking_run.splitter_version,
        "memo": chunking_run.memo,
    }

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(cabinet_qry, {"cabinet_uuid": cabinet_uuid})
        cabinet_row = await cursor.fetchone()
        if cabinet_row is None:
            return False, None

        await cursor.execute(insert_qry, params)
        await cursor.execute(select_qry, {"cabinet_uuid": cabinet_uuid})
        row = await cursor.fetchone()

    if not row:
        return True, None
    return True, ChunkingRun(**row)


_DOCUMENTS_CABINET_FILTER: tuple[str, str] | None = None
_DOCUMENTS_COLUMNS: dict[str, dict[str, str | None]] | None = None
_CABINETS_COLUMNS: dict[str, dict[str, str | None]] | None = None


async def _get_cabinets_columns_async(
    db,
) -> dict[str, dict[str, str | None]]:
    import aiomysql

    global _CABINETS_COLUMNS
    if _CABINETS_COLUMNS is not None:
        return _CABINETS_COLUMNS

    qry = """
    SELECT
        COLUMN_NAME AS column_name,
        IS_NULLABLE AS is_nullable,
        COLUMN_DEFAULT AS column_default,
        EXTRA AS extra,
        COLUMN_TYPE AS column_type
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name = 'cabinets'
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry)
        rows = await cursor.fetchall()

    _CABINETS_COLUMNS = {
        row["column_name"]: {
            "is_nullable": row["is_nullable"],
            "column_default": row["column_default"],
            "extra": row["extra"],
            "column_type": row["column_type"],
        }
        for row in rows
    }
    return _CABINETS_COLUMNS


async def _resolve_documents_cabinet_filter_async(
    db,
) -> tuple[str, str]:
    import aiomysql

    global _DOCUMENTS_CABINET_FILTER
    if _DOCUMENTS_CABINET_FILTER is not None:
        return _DOCUMENTS_CABINET_FILTER

    qry = """
    SELECT COLUMN_NAME AS column_name
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name = 'documents'
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry)
        rows = await cursor.fetchall()
    columns = {row["column_name"] for row in rows}

    if "cabinet_uuid" in columns:
        _DOCUMENTS_CABINET_FILTER = ("", "WHERE d.cabinet_uuid = %(cabinet_uuid)s")
    elif "cabinet_id" in columns:
        _DOCUMENTS_CABINET_FILTER = (
            "JOIN cabinets c ON d.cabinet_id = c.id",
            "WHERE c.cabinet_uuid = %(cabinet_uuid)s",
        )
    else:
        raise RuntimeError(
            "documents table is missing cabinet_uuid/cabinet_id columns"
        )

    return _DOCUMENTS_CABINET_FILTER
