def say_hello(name: str) -> str:
    return f"Hello {name}"


def health_status() -> dict:
    return {"status": "ok"}


async def fetch_models_async(
    db,
    model_type: str | None,
    status_filter: str | None = None,
) -> list["ModelItem"]:
    import aiomysql

    from src.model.kl_models import ModelItem

    where_clauses: list[str] = []
    params: dict[str, object] = {}
    if model_type:
        params["model_type"] = model_type
        where_clauses.append("model_type = %(model_type)s")
    if status_filter == "active":
        where_clauses.append("is_deprecated = 0")
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    qry = """
    SELECT
        id,
        model_type,
        provider,
        model_name,
        model_version,
        dimension,
        is_deprecated,
        created_at
    FROM models
    {where_sql}
    ORDER BY id
    LIMIT 0, 1000
    """.format(where_sql=where_sql)
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, params)
        rows = await cursor.fetchall()
    return [
        ModelItem(
            id=row["id"],
            model_type=row["model_type"],
            provider=row["provider"],
            model_name=row["model_name"],
            model_version=row["model_version"],
            dimension=row["dimension"],
            is_deprecated=bool(row["is_deprecated"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]


async def fetch_models_summary_async(db) -> dict[str, int]:
    import aiomysql

    counts_qry = """
    SELECT
        (SELECT COUNT(*) FROM models) AS models_count,
        (SELECT COUNT(*) FROM llm_model_configs) AS llm_model_configs_count,
        (SELECT COUNT(*) FROM embedding_model_configs) AS embedding_model_configs_count,
        (SELECT COUNT(*) FROM embedding_runs) AS embedding_runs_count,
        (SELECT COUNT(*) FROM system_profiles) AS system_profile_count
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(counts_qry)
        row = await cursor.fetchone()
    row = row or {}
    return {
        "models_count": int(row.get("models_count") or 0),
        "llm_model_configs_count": int(row.get("llm_model_configs_count") or 0),
        "embedding_model_configs_count": int(
            row.get("embedding_model_configs_count") or 0
        ),
        "embedding_runs_count": int(row.get("embedding_runs_count") or 0),
        "system_profile_count": int(row.get("system_profile_count") or 0),
    }


def _extract_prefixed_row(
    row: dict[str, object], prefix: str
) -> dict[str, object]:
    return {
        key[len(prefix) :]: value
        for key, value in row.items()
        if key.startswith(prefix)
    }


def _clean_optional_row(data: dict[str, object]) -> dict[str, object] | None:
    if not data:
        return None
    if data.get("id") is None and all(value is None for value in data.values()):
        return None
    return data


def _build_system_profile_from_row(
    row: dict[str, object],
) -> "SystemProfileItem | None":
    from src.model.kl_models import SystemProfileItem

    sp_data = _extract_prefixed_row(row, "sp_")
    if not sp_data or sp_data.get("id") is None:
        return None
    llm_config = _clean_optional_row(_extract_prefixed_row(row, "llmc_"))
    embedding_config = _clean_optional_row(_extract_prefixed_row(row, "emc_"))
    llm_model = _clean_optional_row(_extract_prefixed_row(row, "llmm_"))
    embedding_model = _clean_optional_row(_extract_prefixed_row(row, "emmm_"))

    return SystemProfileItem(
        id=sp_data.get("id"),
        name=sp_data.get("name"),
        description=sp_data.get("description"),
        llm_model_config_id=sp_data.get("llm_model_config_id"),
        embedding_model_config_id=sp_data.get("embedding_model_config_id"),
        is_active=(
            bool(sp_data.get("is_active"))
            if sp_data.get("is_active") is not None
            else None
        ),
        created_at=sp_data.get("created_at"),
        llm_config=llm_config,
        embedding_config=embedding_config,
        llm_model=llm_model,
        embedding_model=embedding_model,
    )


async def fetch_system_profiles_async(
    db,
    is_active: bool | None = None,
    system_profile_id: int | None = None,
) -> list["SystemProfileItem"]:
    import aiomysql

    sp_columns = await _get_table_columns_async(db, "system_profiles")
    llm_config_columns = await _get_table_columns_async(db, "llm_model_configs")
    embedding_config_columns = await _get_table_columns_async(
        db, "embedding_model_configs"
    )
    model_columns = await _get_table_columns_async(db, "models")

    sp_select = [f"sp.{col} AS sp_{col}" for col in sp_columns]
    llm_config_select = [
        f"llmc.{col} AS llmc_{col}" for col in llm_config_columns
    ]
    embedding_config_select = [
        f"emc.{col} AS emc_{col}" for col in embedding_config_columns
    ]
    llm_model_select = [f"llmm.{col} AS llmm_{col}" for col in model_columns]
    embedding_model_select = [
        f"emmm.{col} AS emmm_{col}" for col in model_columns
    ]
    select_sql = ", ".join(
        sp_select
        + llm_config_select
        + embedding_config_select
        + llm_model_select
        + embedding_model_select
    )

    where_clauses: list[str] = []
    params: dict[str, object] = {}
    if system_profile_id is not None:
        where_clauses.append("sp.id = %(system_profile_id)s")
        params["system_profile_id"] = int(system_profile_id)
    if is_active is not None and "is_active" in sp_columns:
        where_clauses.append("sp.is_active = %(is_active)s")
        params["is_active"] = int(is_active)
    where_sql = (
        f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    )

    qry = f"""
    SELECT {select_sql}
    FROM system_profiles
    AS sp
    LEFT JOIN llm_model_configs llmc ON llmc.id = sp.llm_model_config_id
    LEFT JOIN embedding_model_configs emc ON emc.id = sp.embedding_model_config_id
    LEFT JOIN models llmm ON llmm.id = llmc.model_id
    LEFT JOIN models emmm ON emmm.id = emc.model_id
    {where_sql}
    ORDER BY sp.id
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, params)
        rows = await cursor.fetchall()

    return [
        item
        for row in rows
        if (item := _build_system_profile_from_row(row)) is not None
    ]


async def create_system_profile_async(
    db,
    payload: "SystemProfileCreateRequest",
) -> "SystemProfileItem":
    import aiomysql

    from src.model.kl_models import SystemProfileCreateRequest

    if not isinstance(payload, SystemProfileCreateRequest):
        raise TypeError("payload must be SystemProfileCreateRequest")

    columns = await _get_table_columns_async(db, "system_profiles")
    insert_fields = []
    params: dict[str, object] = {}
    values = {
        "name": payload.name,
        "description": payload.description,
        "llm_model_config_id": payload.llm_model_config_id,
        "embedding_model_config_id": payload.embedding_model_config_id,
        "is_active": int(payload.is_active),
    }
    for key, value in values.items():
        if key in columns:
            insert_fields.append(key)
            params[key] = value
    if "created_at" in columns:
        insert_fields.append("created_at")
        params["created_at"] = None
    if "updated_at" in columns:
        insert_fields.append("updated_at")
        params["updated_at"] = None

    if not insert_fields:
        raise RuntimeError("system_profiles has no writable columns")

    value_placeholders = []
    for field in insert_fields:
        value_placeholders.append(
            "NOW()" if field in ("created_at", "updated_at") else f"%({field})s"
        )

    insert_qry = (
        "INSERT INTO system_profiles ("
        + ", ".join(insert_fields)
        + ") VALUES ("
        + ", ".join(value_placeholders)
        + ")"
    )

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(insert_qry, params)
        system_profile_id = cursor.lastrowid

    items = await fetch_system_profiles_async(
        db, system_profile_id=system_profile_id
    )
    if not items:
        raise RuntimeError("Failed to load created system profile")
    return items[0]


async def delete_system_profile_async(
    db,
    system_profile_id: int,
) -> bool:
    import aiomysql

    delete_qry = """
    DELETE FROM system_profiles
    WHERE id = %(id)s
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(delete_qry, {"id": system_profile_id})
        return cursor.rowcount > 0


async def update_system_profile_async(
    db,
    payload: "SystemProfileUpdateRequest",
) -> tuple[bool, "SystemProfileItem | None"]:
    import aiomysql

    from src.model.kl_models import SystemProfileUpdateRequest

    if not isinstance(payload, SystemProfileUpdateRequest):
        raise TypeError("payload must be SystemProfileUpdateRequest")

    columns = await _get_table_columns_async(db, "system_profiles")
    exists_qry = "SELECT id FROM system_profiles WHERE id = %(id)s LIMIT 1"
    update_fields = []
    params: dict[str, object] = {"id": payload.id}
    values = {
        "name": payload.name,
        "description": payload.description,
        "llm_model_config_id": payload.llm_model_config_id,
        "embedding_model_config_id": payload.embedding_model_config_id,
        "is_active": int(payload.is_active),
    }
    for key, value in values.items():
        if key in columns:
            update_fields.append(f"{key} = %({key})s")
            params[key] = value
    if "updated_at" in columns:
        update_fields.append("updated_at = NOW()")

    if not update_fields:
        raise RuntimeError("system_profiles has no writable columns")

    update_qry = (
        "UPDATE system_profiles SET "
        + ", ".join(update_fields)
        + " WHERE id = %(id)s"
    )

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, {"id": payload.id})
        exists = await cursor.fetchone()
        if exists is None:
            return False, None
        await cursor.execute(update_qry, params)

    items = await fetch_system_profiles_async(
        db, system_profile_id=payload.id
    )
    if not items:
        return True, None
    return True, items[0]


async def deactivate_system_profile_async(
    db,
    system_profile_id: int,
) -> tuple[bool, "SystemProfileItem | None", list[dict[str, object]] | None, str | None]:
    import aiomysql

    sp_columns = await _get_table_columns_async(db, "system_profiles")
    if "is_active" not in sp_columns:
        raise RuntimeError("system_profiles table missing is_active")

    cabinet_columns = await _get_cabinets_columns_async(db)
    if "system_profile_id" not in cabinet_columns:
        raise RuntimeError("cabinets table missing system_profile_id")
    if "is_active" not in cabinet_columns:
        raise RuntimeError("cabinets table missing is_active")

    exists_qry = "SELECT id FROM system_profiles WHERE id = %(id)s LIMIT 1"
    cabinets_qry = """
    SELECT
        id,
        project_id,
        cabinet_uuid,
        cabinet_name
    FROM cabinets
    WHERE system_profile_id = %(id)s
      AND is_active = 1
    ORDER BY project_id, cabinet_name, cabinet_uuid
    """
    update_qry = """
    UPDATE system_profiles
    SET is_active = 0
    WHERE id = %(id)s
    """

    params = {"id": system_profile_id}
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, params)
        exists = await cursor.fetchone()
        if exists is None:
            return False, None, None, "not_found"

        await cursor.execute(cabinets_qry, params)
        cabinets = await cursor.fetchall() or []
        if cabinets:
            cabinet_items = [
                {
                    "id": row["id"],
                    "project_id": row["project_id"],
                    "cabinet_uuid": row["cabinet_uuid"],
                    "name": row["cabinet_name"],
                }
                for row in cabinets
            ]
            return True, None, cabinet_items, "has_active_cabinets"

        await cursor.execute(update_qry, params)

    items = await fetch_system_profiles_async(
        db, system_profile_id=system_profile_id
    )
    if not items:
        return True, None, None, "update_failed"
    return True, items[0], None, None


async def activate_system_profile_async(
    db,
    system_profile_id: int,
) -> tuple[bool, "SystemProfileItem | None", str | None]:
    import aiomysql

    sp_columns = await _get_table_columns_async(db, "system_profiles")
    if "is_active" not in sp_columns:
        raise RuntimeError("system_profiles table missing is_active")

    exists_qry = "SELECT id FROM system_profiles WHERE id = %(id)s LIMIT 1"
    update_qry = """
    UPDATE system_profiles
    SET is_active = 1
    WHERE id = %(id)s
    """
    params = {"id": system_profile_id}

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, params)
        exists = await cursor.fetchone()
        if exists is None:
            return False, None, "not_found"
        await cursor.execute(update_qry, params)

    items = await fetch_system_profiles_async(
        db, system_profile_id=system_profile_id
    )
    if not items:
        return True, None, "update_failed"
    return True, items[0], None


async def update_model_deprecated_async(
    db,
    payload: "ModelUpdateRequest",
) -> tuple[bool, "ModelItem | None"]:
    import aiomysql

    from src.model.kl_models import ModelItem, ModelUpdateRequest

    if not isinstance(payload, ModelUpdateRequest):
        raise TypeError("payload must be ModelUpdateRequest")

    exists_qry = "SELECT id FROM models WHERE id = %(id)s LIMIT 1"
    update_qry = """
    UPDATE models
    SET is_deprecated = %(is_deprecated)s
    WHERE id = %(id)s
    """
    select_qry = """
    SELECT
        id,
        model_type,
        provider,
        model_name,
        model_version,
        dimension,
        is_deprecated,
        created_at
    FROM models
    WHERE id = %(id)s
    LIMIT 1
    """
    params = {"id": payload.id, "is_deprecated": int(payload.is_deprecated)}
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, {"id": payload.id})
        exists = await cursor.fetchone()
        if exists is None:
            return False, None
        await cursor.execute(update_qry, params)
        await cursor.execute(select_qry, {"id": payload.id})
        row = await cursor.fetchone()
    if row is None:
        return True, None
    return True, ModelItem(
        id=row["id"],
        model_type=row["model_type"],
        provider=row["provider"],
        model_name=row["model_name"],
        model_version=row["model_version"],
        dimension=row["dimension"],
        is_deprecated=bool(row["is_deprecated"]),
        created_at=row["created_at"],
    )


async def fetch_model_configs_async(
    db,
    model_id: int | None,
    model_type: str | None,
) -> tuple[bool, str | None, list[dict[str, object]] | None, dict[str, list[dict[str, object]]] | None]:
    import aiomysql

    async def _fetch_configs(
        table: str, model_id_value: int | None
    ) -> list[dict[str, object]]:
        columns = await _get_table_columns_async(db, table)
        order_sql = " ORDER BY id DESC" if "id" in columns else ""
        where_clauses: list[str] = []
        params: dict[str, object] = {}
        if model_id_value is not None:
            model_id_col = _pick_first_column(columns, ("model_id",))
            if not model_id_col:
                raise RuntimeError(f"{table} missing model_id column")
            where_clauses.append(f"{model_id_col} = %(model_id)s")
            params["model_id"] = model_id_value
        where_sql = (
            f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        )
        select_qry = f"SELECT * FROM {table}{where_sql}{order_sql}"
        async with db.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(select_qry, params)
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    if model_id is not None:
        qry = """
        SELECT id, model_type
        FROM models
        WHERE id = %(id)s
        LIMIT 1
        """
        async with db.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(qry, {"id": model_id})
            row = await cursor.fetchone()
        if row is None:
            return False, None, None, None

        model_type_value = row.get("model_type")
        if not model_type_value:
            return True, None, None, None

        normalized = str(model_type_value).upper()
        if normalized == "LLM":
            table = "llm_model_configs"
        elif normalized == "EMBEDDING":
            table = "embedding_model_configs"
        else:
            return True, normalized, None, None

        configs = await _fetch_configs(table, model_id)
        return True, normalized, configs or None, None

    if model_type is None:
        return True, None, None, None

    normalized = str(model_type).strip().upper()
    if normalized == "LLM":
        configs = await _fetch_configs("llm_model_configs", None)
        return True, normalized, configs or None, None
    if normalized == "EMBEDDING":
        configs = await _fetch_configs("embedding_model_configs", None)
        return True, normalized, configs or None, None
    if normalized == "ALL":
        llm_configs = await _fetch_configs("llm_model_configs", None)
        embedding_configs = await _fetch_configs("embedding_model_configs", None)
        return True, normalized, None, {
            "llm_configs": llm_configs,
            "embedding_configs": embedding_configs,
        }
    return True, normalized, None, None


async def create_llm_model_config_async(
    db,
    payload: "LLMModelConfigCreateRequest",
) -> dict[str, object]:
    import aiomysql

    from src.model.kl_models import LLMModelConfigCreateRequest

    if not isinstance(payload, LLMModelConfigCreateRequest):
        raise TypeError("payload must be LLMModelConfigCreateRequest")

    columns = await _get_table_columns_async(db, "llm_model_configs")
    insert_fields = []
    params: dict[str, object] = {}
    values = {
        "model_id": payload.model_id,
        "temperature": payload.temperature,
        "top_p": payload.top_p,
        "max_tokens": payload.max_tokens,
        "system_prompt": payload.system_prompt,
    }
    for key, value in values.items():
        if key in columns:
            insert_fields.append(key)
            params[key] = value
    if "created_at" in columns:
        insert_fields.append("created_at")
        params["created_at"] = None
    if "updated_at" in columns:
        insert_fields.append("updated_at")
        params["updated_at"] = None

    if not insert_fields:
        raise RuntimeError("llm_model_configs has no writable columns")

    value_placeholders = []
    for field in insert_fields:
        value_placeholders.append(
            "NOW()" if field in ("created_at", "updated_at") else f"%({field})s"
        )

    insert_qry = (
        "INSERT INTO llm_model_configs ("
        + ", ".join(insert_fields)
        + ") VALUES ("
        + ", ".join(value_placeholders)
        + ")"
    )
    select_qry = "SELECT * FROM llm_model_configs WHERE id = %(id)s LIMIT 1"

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(insert_qry, params)
        config_id = cursor.lastrowid
        await cursor.execute(select_qry, {"id": config_id})
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("Failed to load created llm_model_configs")
    return dict(row)


async def create_embedding_model_config_async(
    db,
    payload: "EmbeddingModelConfigCreateRequest",
) -> dict[str, object]:
    import aiomysql

    from src.model.kl_models import EmbeddingModelConfigCreateRequest

    if not isinstance(payload, EmbeddingModelConfigCreateRequest):
        raise TypeError("payload must be EmbeddingModelConfigCreateRequest")

    columns = await _get_table_columns_async(db, "embedding_model_configs")
    insert_fields = []
    params: dict[str, object] = {}
    values = {
        "model_id": payload.model_id,
        "chunk_size": payload.chunk_size,
        "overlap": payload.overlap,
        "normalize": payload.normalize,
        "distance_metric": payload.distance_metric,
        "length_unit": payload.length_unit,
    }
    for key, value in values.items():
        if key in columns:
            insert_fields.append(key)
            params[key] = value
    if "created_at" in columns:
        insert_fields.append("created_at")
        params["created_at"] = None
    if "updated_at" in columns:
        insert_fields.append("updated_at")
        params["updated_at"] = None

    if not insert_fields:
        raise RuntimeError("embedding_model_configs has no writable columns")

    value_placeholders = []
    for field in insert_fields:
        value_placeholders.append(
            "NOW()" if field in ("created_at", "updated_at") else f"%({field})s"
        )

    insert_qry = (
        "INSERT INTO embedding_model_configs ("
        + ", ".join(insert_fields)
        + ") VALUES ("
        + ", ".join(value_placeholders)
        + ")"
    )
    select_qry = (
        "SELECT * FROM embedding_model_configs WHERE id = %(id)s LIMIT 1"
    )

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(insert_qry, params)
        config_id = cursor.lastrowid
        await cursor.execute(select_qry, {"id": config_id})
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("Failed to load created embedding_model_configs")
    return dict(row)


async def update_llm_model_config_async(
    db,
    payload: "LLMModelConfigCreateRequest",
) -> tuple[bool, dict[str, object] | None]:
    import aiomysql

    from src.model.kl_models import LLMModelConfigCreateRequest

    if not isinstance(payload, LLMModelConfigCreateRequest):
        raise TypeError("payload must be LLMModelConfigCreateRequest")

    columns = await _get_table_columns_async(db, "llm_model_configs")
    model_id_col = _pick_first_column(columns, ("model_id",))
    if not model_id_col:
        raise RuntimeError("llm_model_configs missing model_id column")

    exists_qry = (
        f"SELECT id FROM llm_model_configs WHERE {model_id_col} = %(model_id)s"
        " LIMIT 1"
    )
    update_fields = []
    params: dict[str, object] = {"model_id": payload.model_id}
    values = {
        "temperature": payload.temperature,
        "top_p": payload.top_p,
        "max_tokens": payload.max_tokens,
        "system_prompt": payload.system_prompt,
    }
    for key, value in values.items():
        if key in columns:
            update_fields.append(f"{key} = %({key})s")
            params[key] = value
    if "updated_at" in columns:
        update_fields.append("updated_at = NOW()")

    if not update_fields:
        raise RuntimeError("llm_model_configs has no writable columns")

    update_qry = (
        f"UPDATE llm_model_configs SET {', '.join(update_fields)}"
        f" WHERE {model_id_col} = %(model_id)s"
    )
    order_sql = " ORDER BY id DESC" if "id" in columns else ""
    select_qry = (
        f"SELECT * FROM llm_model_configs WHERE {model_id_col} = %(model_id)s"
        f"{order_sql} LIMIT 1"
    )

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, {"model_id": payload.model_id})
        exists = await cursor.fetchone()
        if exists is None:
            return False, None
        await cursor.execute(update_qry, params)
        await cursor.execute(select_qry, {"model_id": payload.model_id})
        row = await cursor.fetchone()
    if row is None:
        return True, None
    return True, dict(row)


async def update_embedding_model_config_async(
    db,
    payload: "EmbeddingModelConfigCreateRequest",
) -> tuple[bool, dict[str, object] | None]:
    import aiomysql

    from src.model.kl_models import EmbeddingModelConfigCreateRequest

    if not isinstance(payload, EmbeddingModelConfigCreateRequest):
        raise TypeError("payload must be EmbeddingModelConfigCreateRequest")

    columns = await _get_table_columns_async(db, "embedding_model_configs")
    model_id_col = _pick_first_column(columns, ("model_id",))
    if not model_id_col:
        raise RuntimeError("embedding_model_configs missing model_id column")

    exists_qry = (
        f"SELECT id FROM embedding_model_configs WHERE {model_id_col} = %(model_id)s"
        " LIMIT 1"
    )
    update_fields = []
    params: dict[str, object] = {"model_id": payload.model_id}
    values = {
        "chunk_size": payload.chunk_size,
        "overlap": payload.overlap,
        "normalize": payload.normalize,
        "distance_metric": payload.distance_metric,
        "length_unit": payload.length_unit,
    }
    for key, value in values.items():
        if key in columns:
            update_fields.append(f"{key} = %({key})s")
            params[key] = value
    if "updated_at" in columns:
        update_fields.append("updated_at = NOW()")

    if not update_fields:
        raise RuntimeError("embedding_model_configs has no writable columns")

    update_qry = (
        f"UPDATE embedding_model_configs SET {', '.join(update_fields)}"
        f" WHERE {model_id_col} = %(model_id)s"
    )
    order_sql = " ORDER BY id DESC" if "id" in columns else ""
    select_qry = (
        f"SELECT * FROM embedding_model_configs"
        f" WHERE {model_id_col} = %(model_id)s{order_sql} LIMIT 1"
    )

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, {"model_id": payload.model_id})
        exists = await cursor.fetchone()
        if exists is None:
            return False, None
        await cursor.execute(update_qry, params)
        await cursor.execute(select_qry, {"model_id": payload.model_id})
        row = await cursor.fetchone()
    if row is None:
        return True, None
    return True, dict(row)


async def delete_llm_model_config_async(
    db,
    config_id: int,
) -> bool:
    import aiomysql

    exists_qry = "SELECT id FROM llm_model_configs WHERE id = %(id)s LIMIT 1"
    delete_qry = "DELETE FROM llm_model_configs WHERE id = %(id)s"
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, {"id": config_id})
        row = await cursor.fetchone()
        if row is None:
            return False
        await cursor.execute(delete_qry, {"id": config_id})
    return True


async def delete_embedding_model_config_async(
    db,
    config_id: int,
) -> bool:
    import aiomysql

    exists_qry = (
        "SELECT id FROM embedding_model_configs WHERE id = %(id)s LIMIT 1"
    )
    delete_qry = "DELETE FROM embedding_model_configs WHERE id = %(id)s"
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, {"id": config_id})
        row = await cursor.fetchone()
        if row is None:
            return False
        await cursor.execute(delete_qry, {"id": config_id})
    return True


async def fetch_storage_types_async(db) -> list[str]:
    import aiomysql

    qry = """
    SELECT storage_type
    FROM storages
    ORDER BY storage_type
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry)
        rows = await cursor.fetchall()
    return [row["storage_type"] for row in rows]


async def fetch_vector_stores_async(db) -> list[str]:
    import aiomysql

    qry = """
    SELECT vector_store
    FROM vector_stores
    ORDER BY vector_store
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry)
        rows = await cursor.fetchall()
    return [row["vector_store"] for row in rows]


async def fetch_system_secret_async(
    db,
    key: str,
    provider: str,
    env: str,
    usage_scope: str,
) -> str | None:
    import aiomysql

    qry = """
    SELECT secret_value
    FROM system_secrets
    WHERE provider = %(provider)s
      AND secret_name = %(key)s
      AND env = %(env)s
      AND usage_scope = %(usage_scope)s
    LIMIT 1
    """
    params = {
        "provider": provider,
        "key": key,
        "env": env,
        "usage_scope": usage_scope,
    }
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, params)
        row = await cursor.fetchone()
    if row is None:
        return None
    return row["secret_value"]


def _mask_secret_value(value: str) -> str:
    return "*" * 32


async def fetch_system_secrets_async(
    db,
    env: str | None,
) -> tuple[list["SystemSecretListItem"], dict[str, int]]:
    import aiomysql

    from src.model.kl_models import SystemSecretListItem

    where_sql = ""
    params: dict[str, object] = {}
    if env:
        where_sql = "WHERE env = %(env)s"
        params["env"] = env

    qry = """
    SELECT
        id,
        provider,
        secret_name,
        secret_value,
        usage_scope,
        env,
        is_active,
        rotated_at,
        created_at,
        updated_at
    FROM system_secrets
    {where_sql}
    ORDER BY id DESC
    """.format(where_sql=where_sql)
    stats_qry = """
    SELECT
        COUNT(*) AS total_count,
        SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active_count,
        SUM(CASE WHEN is_active = 0 THEN 1 ELSE 0 END) AS inactive_count,
        SUM(
            CASE
                WHEN rotated_at IS NOT NULL
                 AND rotated_at >= (NOW() - INTERVAL 7 DAY)
                THEN 1 ELSE 0
            END
        ) AS rotated_last_7_days_count
    FROM system_secrets
    {where_sql}
    """.format(where_sql=where_sql)
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, params)
        rows = await cursor.fetchall()
        await cursor.execute(stats_qry, params)
        stats_row = await cursor.fetchone()
    items = [
        SystemSecretListItem(
            id=row["id"],
            provider=row["provider"],
            secret_name=row["secret_name"],
            secret_value=_mask_secret_value(row["secret_value"] or ""),
            usage_scope=row["usage_scope"],
            env=row["env"],
            is_active=bool(row["is_active"]),
            rotated_at=row["rotated_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]
    stats_row = stats_row or {}
    stats = {
        "total_count": int(stats_row.get("total_count") or 0),
        "active_count": int(stats_row.get("active_count") or 0),
        "inactive_count": int(stats_row.get("inactive_count") or 0),
        "rotated_last_7_days_count": int(
            stats_row.get("rotated_last_7_days_count") or 0
        ),
    }
    return items, stats


async def create_system_secret_async(
    db,
    payload: "SystemSecretCreateRequest",
) -> "SystemSecretListItem":
    import aiomysql

    from src.model.kl_models import SystemSecretCreateRequest, SystemSecretListItem

    if not isinstance(payload, SystemSecretCreateRequest):
        raise TypeError("payload must be SystemSecretCreateRequest")

    insert_qry = """
    INSERT INTO system_secrets (
        provider,
        secret_name,
        secret_value,
        usage_scope,
        env
    ) VALUES (
        %(provider)s,
        %(secret_name)s,
        %(secret_value)s,
        %(usage_scope)s,
        %(env)s
    )
    """
    select_qry = """
    SELECT
        id,
        provider,
        secret_name,
        secret_value,
        usage_scope,
        env,
        is_active,
        rotated_at,
        created_at,
        updated_at
    FROM system_secrets
    WHERE id = %(id)s
    LIMIT 1
    """
    params = {
        "provider": payload.provider,
        "secret_name": payload.secret_name,
        "secret_value": payload.secret_value,
        "usage_scope": payload.usage_scope,
        "env": payload.env,
    }
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(insert_qry, params)
        secret_id = cursor.lastrowid
        await cursor.execute(select_qry, {"id": secret_id})
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("Failed to load created system secret")
    return SystemSecretListItem(
        id=row["id"],
        provider=row["provider"],
        secret_name=row["secret_name"],
        secret_value=_mask_secret_value(row["secret_value"] or ""),
        usage_scope=row["usage_scope"],
        env=row["env"],
        is_active=bool(row["is_active"]),
        rotated_at=row["rotated_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def rotate_system_secret_async(
    db,
    secret_id: int,
    secret_value: str,
) -> "SystemSecretListItem | None":
    import aiomysql

    select_qry = """
    SELECT
        id,
        provider,
        secret_name,
        secret_value,
        usage_scope,
        env,
        is_active,
        rotated_at,
        created_at,
        updated_at
    FROM system_secrets
    WHERE id = %(id)s
    LIMIT 1
    """
    update_qry = """
    UPDATE system_secrets
    SET
        secret_value = %(secret_value)s,
        rotated_at = NOW()
    WHERE id = %(id)s
    """
    params = {"id": secret_id, "secret_value": secret_value}
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(update_qry, params)
        if cursor.rowcount == 0:
            return None
        await cursor.execute(select_qry, {"id": secret_id})
        row = await cursor.fetchone()
    if row is None:
        return None
    from src.model.kl_models import SystemSecretListItem

    return SystemSecretListItem(
        id=row["id"],
        provider=row["provider"],
        secret_name=row["secret_name"],
        secret_value=_mask_secret_value(row["secret_value"] or ""),
        usage_scope=row["usage_scope"],
        env=row["env"],
        is_active=bool(row["is_active"]),
        rotated_at=row["rotated_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def delete_system_secret_async(
    db,
    secret_id: int,
) -> bool:
    import aiomysql

    delete_qry = """
    DELETE FROM system_secrets
    WHERE id = %(id)s
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(delete_qry, {"id": secret_id})
        return cursor.rowcount > 0


async def fetch_system_environments_async(db) -> list[str]:
    import aiomysql

    qry = """
    SELECT env
    FROM system_environments
    ORDER BY `order`
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry)
        rows = await cursor.fetchall()
    return [row["env"] for row in rows]


async def fetch_system_secret_providers_async(db) -> list[str]:
    import aiomysql

    qry = """
    SELECT provider
    FROM system_secret_providers
    WHERE is_active = 1
    ORDER BY `order` ASC, provider ASC
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry)
        rows = await cursor.fetchall()
    return [row["provider"] for row in rows]


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
        id,
        project_name,
        project_owner,
        project_memo,
        is_active,
        created_at,
        updated_at
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
            is_active=bool(row["is_active"]) if row["is_active"] is not None else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


async def create_project_async(
    db,
    payload: "ProjectCreateRequest",
) -> "Project":
    import aiomysql

    from src.model.kl_models import Project, ProjectCreateRequest

    if not isinstance(payload, ProjectCreateRequest):
        raise TypeError("payload must be ProjectCreateRequest")

    insert_qry = """
    INSERT INTO projects (
        project_name,
        project_owner,
        project_memo,
        is_active
    ) VALUES (
        %(project_name)s,
        %(project_owner)s,
        %(project_memo)s,
        %(is_active)s
    )
    """
    select_qry = """
    SELECT
        id,
        project_name,
        project_owner,
        project_memo,
        is_active,
        created_at,
        updated_at
    FROM projects
    WHERE id = %(id)s
    LIMIT 1
    """
    params = {
        "project_name": payload.project_name,
        "project_owner": payload.project_owner,
        "project_memo": payload.project_memo,
        "is_active": int(payload.is_active),
    }
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(insert_qry, params)
        project_id = cursor.lastrowid
        await cursor.execute(select_qry, {"id": project_id})
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("Failed to load created project")
    return Project(
        id=row["id"],
        name=row["project_name"],
        owner=row["project_owner"],
        memo=row["project_memo"],
        is_active=bool(row["is_active"]) if row["is_active"] is not None else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def delete_project_async(db, project_id: int) -> tuple[bool, str | None]:
    import aiomysql

    exists_qry = "SELECT id FROM projects WHERE id = %(id)s LIMIT 1"
    cabinets_qry = """
    SELECT COUNT(*) AS cabinet_count
    FROM cabinets
    WHERE project_id = %(id)s
    """
    documents_qry = """
    SELECT COUNT(*) AS document_count
    FROM documents d
    JOIN cabinets c ON d.cabinet_uuid = c.cabinet_uuid
    WHERE c.project_id = %(id)s
    """
    delete_qry = "DELETE FROM projects WHERE id = %(id)s"
    params = {"id": project_id}

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, params)
        exists = await cursor.fetchone()
        if exists is None:
            return False, "not_found"

        await cursor.execute(cabinets_qry, params)
        cabinets_row = await cursor.fetchone() or {}
        if int(cabinets_row.get("cabinet_count") or 0) > 0:
            return False, "has_cabinets"

        await cursor.execute(documents_qry, params)
        documents_row = await cursor.fetchone() or {}
        if int(documents_row.get("document_count") or 0) > 0:
            return False, "has_documents"

        await cursor.execute(delete_qry, params)
    return True, None


async def update_project_async(
    db,
    payload: "ProjectUpdateRequest",
) -> tuple[bool, "Project | None"]:
    import aiomysql

    from src.model.kl_models import Project, ProjectUpdateRequest

    if not isinstance(payload, ProjectUpdateRequest):
        raise TypeError("payload must be ProjectUpdateRequest")

    exists_qry = "SELECT id FROM projects WHERE id = %(id)s LIMIT 1"
    update_qry = """
    UPDATE projects
    SET
        project_name = %(project_name)s,
        project_owner = %(project_owner)s,
        project_memo = %(project_memo)s,
        is_active = %(is_active)s,
        updated_at = NOW()
    WHERE id = %(id)s
    """
    select_qry = """
    SELECT
        id,
        project_name,
        project_owner,
        project_memo,
        is_active,
        created_at,
        updated_at
    FROM projects
    WHERE id = %(id)s
    LIMIT 1
    """
    params = {
        "id": payload.id,
        "project_name": payload.project_name,
        "project_owner": payload.project_owner,
        "project_memo": payload.project_memo,
        "is_active": int(payload.is_active),
    }
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, {"id": payload.id})
        exists = await cursor.fetchone()
        if exists is None:
            return False, None
        await cursor.execute(update_qry, params)
        await cursor.execute(select_qry, {"id": payload.id})
        row = await cursor.fetchone()
    if row is None:
        return True, None
    return True, Project(
        id=row["id"],
        name=row["project_name"],
        owner=row["project_owner"],
        memo=row["project_memo"],
        is_active=bool(row["is_active"]) if row["is_active"] is not None else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def fetch_projects_summary_async(
    db,
) -> tuple[dict[str, int], list["Project"]]:
    import aiomysql

    from src.model.kl_models import Project

    summary_qry = """
    SELECT
        COUNT(*) AS total_projects,
        SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active_projects
    FROM projects
    """
    cabinets_qry = """
    SELECT COUNT(*) AS total_cabinets
    FROM cabinets
    """
    documents_qry = """
    SELECT COUNT(*) AS total_documents
    FROM documents
    """
    list_qry = """
    SELECT
        p.id AS id,
        p.project_name AS project_name,
        p.project_owner AS project_owner,
        p.project_memo AS project_memo,
        p.is_active AS is_active,
        p.created_at AS created_at,
        p.updated_at AS updated_at,
        COUNT(DISTINCT c.id) AS cabinet_count,
        COUNT(DISTINCT d.doc_uuid) AS document_count
    FROM projects p
    LEFT JOIN cabinets c ON c.project_id = p.id
    LEFT JOIN documents d ON d.cabinet_uuid = c.cabinet_uuid
    GROUP BY
        p.id,
        p.project_name,
        p.project_owner,
        p.project_memo,
        p.is_active,
        p.created_at,
        p.updated_at
    ORDER BY p.id
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(summary_qry)
        summary_row = await cursor.fetchone()
        await cursor.execute(cabinets_qry)
        cabinets_row = await cursor.fetchone()
        await cursor.execute(documents_qry)
        documents_row = await cursor.fetchone()
        await cursor.execute(list_qry)
        rows = await cursor.fetchall()

    summary_row = summary_row or {}
    cabinets_row = cabinets_row or {}
    documents_row = documents_row or {}
    summary = {
        "total_projects": int(summary_row.get("total_projects") or 0),
        "active_projects": int(summary_row.get("active_projects") or 0),
        "total_cabinets": int(cabinets_row.get("total_cabinets") or 0),
        "total_documents": int(documents_row.get("total_documents") or 0),
    }
    items = [
        Project(
            id=row["id"],
            name=row["project_name"],
            owner=row["project_owner"],
            memo=row["project_memo"],
            is_active=bool(row["is_active"]) if row["is_active"] is not None else None,
            cabinet_count=int(row["cabinet_count"] or 0),
            document_count=int(row["document_count"] or 0),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]
    return summary, items


async def _build_cabinet_select_sql(db, table_alias: str = "c") -> str:
    storage_cols = await _resolve_cabinet_storage_columns_async(db)
    storage_type_col = storage_cols["storage_type"]
    base_path_col = storage_cols["storage_base_path"]
    upload_path_col = storage_cols["storage_path"]

    sp_columns = await _get_table_columns_async(db, "system_profiles")
    llm_config_columns = await _get_table_columns_async(db, "llm_model_configs")
    embedding_config_columns = await _get_table_columns_async(
        db, "embedding_model_configs"
    )
    model_columns = await _get_table_columns_async(db, "models")

    cabinet_select = [
        f"{table_alias}.id",
        f"{table_alias}.project_id",
        f"{table_alias}.cabinet_uuid",
        f"{table_alias}.cabinet_name",
        f"{table_alias}.{storage_type_col} AS storage_type",
        f"{table_alias}.{base_path_col} AS storage_base_path",
        f"{table_alias}.{upload_path_col} AS storage_path",
        f"{table_alias}.vector_store",
        f"{table_alias}.collection_name",
        f"{table_alias}.is_active",
    ]
    sp_select = [f"sp.{col} AS sp_{col}" for col in sp_columns]
    llm_config_select = [
        f"llmc.{col} AS llmc_{col}" for col in llm_config_columns
    ]
    embedding_config_select = [
        f"emc.{col} AS emc_{col}" for col in embedding_config_columns
    ]
    llm_model_select = [f"llmm.{col} AS llmm_{col}" for col in model_columns]
    embedding_model_select = [
        f"emmm.{col} AS emmm_{col}" for col in model_columns
    ]
    return ", ".join(
        cabinet_select
        + sp_select
        + llm_config_select
        + embedding_config_select
        + llm_model_select
        + embedding_model_select
    )


async def fetch_cabinets_by_project_async(db, project_id: int) -> list["Cabinet"]:
    import aiomysql

    from src.model.kl_models import Cabinet

    select_sql = await _build_cabinet_select_sql(db, table_alias="c")
    qry = f"""
    SELECT {select_sql}
    FROM cabinets c
    LEFT JOIN system_profiles sp ON sp.id = c.system_profile_id
    LEFT JOIN llm_model_configs llmc ON llmc.id = sp.llm_model_config_id
    LEFT JOIN embedding_model_configs emc ON emc.id = sp.embedding_model_config_id
    LEFT JOIN models llmm ON llmm.id = llmc.model_id
    LEFT JOIN models emmm ON emmm.id = emc.model_id
    WHERE c.project_id = %(project_id)s
    ORDER BY c.id
    LIMIT 0 , 1000
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, {"project_id": project_id})
        rows = await cursor.fetchall()
    return [
        Cabinet(
            id=row["id"],
            project_id=row["project_id"],
            cabinet_uuid=row["cabinet_uuid"],
            name=row["cabinet_name"],
            storage_type=row["storage_type"],
            storage_base_path=row["storage_base_path"],
            storage_path=row["storage_path"],
            vector_store=row["vector_store"],
            collection_name=row["collection_name"],
            system_profile=_build_system_profile_from_row(row),
            is_active=bool(row["is_active"]) if row["is_active"] is not None else None,
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

    select_sql = await _build_cabinet_select_sql(db, table_alias="c")
    qry = f"""
    SELECT {select_sql}
    FROM cabinets c
    LEFT JOIN system_profiles sp ON sp.id = c.system_profile_id
    LEFT JOIN llm_model_configs llmc ON llmc.id = sp.llm_model_config_id
    LEFT JOIN embedding_model_configs emc ON emc.id = sp.embedding_model_config_id
    LEFT JOIN models llmm ON llmm.id = llmc.model_id
    LEFT JOIN models emmm ON emmm.id = emc.model_id
    WHERE c.project_id = %(project_id)s
        AND c.cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
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
        cabinet_uuid=row["cabinet_uuid"],
        name=row["cabinet_name"],
        storage_type=row["storage_type"],
        storage_base_path=row["storage_base_path"],
        storage_path=row["storage_path"],
        vector_store=row["vector_store"],
        collection_name=row["collection_name"],
        system_profile=_build_system_profile_from_row(row),
        is_active=bool(row["is_active"]) if row["is_active"] is not None else None,
    )


async def fetch_cabinet_by_uuid_async(
    db,
    cabinet_uuid: str,
) -> "Cabinet | None":
    import aiomysql

    from src.model.kl_models import Cabinet

    select_sql = await _build_cabinet_select_sql(db, table_alias="c")
    qry = f"""
    SELECT {select_sql}
    FROM cabinets c
    LEFT JOIN system_profiles sp ON sp.id = c.system_profile_id
    LEFT JOIN llm_model_configs llmc ON llmc.id = sp.llm_model_config_id
    LEFT JOIN embedding_model_configs emc ON emc.id = sp.embedding_model_config_id
    LEFT JOIN models llmm ON llmm.id = llmc.model_id
    LEFT JOIN models emmm ON emmm.id = emc.model_id
    WHERE c.cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, {"cabinet_uuid": cabinet_uuid})
        row = await cursor.fetchone()
    if row is None:
        return None
    return Cabinet(
        id=row["id"],
        project_id=row["project_id"],
        cabinet_uuid=row["cabinet_uuid"],
        name=row["cabinet_name"],
        storage_type=row["storage_type"],
        storage_base_path=row["storage_base_path"],
        storage_path=row["storage_path"],
        vector_store=row["vector_store"],
        collection_name=row["collection_name"],
        system_profile=_build_system_profile_from_row(row),
        is_active=bool(row["is_active"]) if row["is_active"] is not None else None,
    )


async def update_cabinet_async(
    db,
    payload: "CabinetUpdateRequest",
) -> tuple[bool, "Cabinet | None", str | None]:
    import aiomysql

    from src.model.kl_models import Cabinet, CabinetUpdateRequest

    if not isinstance(payload, CabinetUpdateRequest):
        raise TypeError("payload must be CabinetUpdateRequest")

    storage_cols = await _resolve_cabinet_storage_columns_async(db)
    storage_type_col = storage_cols["storage_type"]
    base_path_col = storage_cols["storage_base_path"]
    upload_path_col = storage_cols["storage_path"]

    columns = await _get_cabinets_columns_async(db)
    if "system_profile_id" not in columns:
        raise RuntimeError("cabinets table missing system_profile_id")

    cabinet_qry = """
    SELECT id
    FROM cabinets
    WHERE project_id = %(project_id)s
      AND cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    update_qry = f"""
    UPDATE cabinets
    SET
        cabinet_name = %(name)s,
        {storage_type_col} = %(storage_type)s,
        {base_path_col} = %(storage_base_path)s,
        {upload_path_col} = %(storage_path)s,
        vector_store = %(vector_store)s,
        collection_name = %(collection_name)s,
        system_profile_id = %(system_profile_id)s,
        is_active = %(is_active)s
    WHERE project_id = %(project_id)s
      AND cabinet_uuid = %(cabinet_uuid)s
    """
    select_sql = await _build_cabinet_select_sql(db, table_alias="c")
    select_qry = f"""
    SELECT {select_sql}
    FROM cabinets c
    LEFT JOIN system_profiles sp ON sp.id = c.system_profile_id
    LEFT JOIN llm_model_configs llmc ON llmc.id = sp.llm_model_config_id
    LEFT JOIN embedding_model_configs emc ON emc.id = sp.embedding_model_config_id
    LEFT JOIN models llmm ON llmm.id = llmc.model_id
    LEFT JOIN models emmm ON emmm.id = emc.model_id
    WHERE c.project_id = %(project_id)s
      AND c.cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    params = {
        "project_id": payload.project_id,
        "cabinet_uuid": payload.cabinet_uuid,
        "name": payload.name,
        "storage_type": payload.storage_type,
        "storage_base_path": payload.storage_base_path,
        "storage_path": payload.storage_path,
        "vector_store": payload.vector_store,
        "collection_name": payload.collection_name,
        "system_profile_id": payload.system_profile_id,
        "is_active": int(payload.is_active),
    }

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(cabinet_qry, params)
        cabinet_row = await cursor.fetchone()
        if cabinet_row is None:
            return False, None, "not_found"
        await cursor.execute(
            "SELECT id FROM system_profiles WHERE id = %(id)s LIMIT 1",
            {"id": payload.system_profile_id},
        )
        sp_row = await cursor.fetchone()
        if sp_row is None:
            return False, None, "system_profile_missing"
        await cursor.execute(update_qry, params)
        await cursor.execute(select_qry, params)
        row = await cursor.fetchone()

    if row is None:
        return True, None, "update_failed"
    return True, Cabinet(
        id=row["id"],
        project_id=row["project_id"],
        cabinet_uuid=row["cabinet_uuid"],
        name=row["cabinet_name"],
        storage_type=row["storage_type"],
        storage_base_path=row["storage_base_path"],
        storage_path=row["storage_path"],
        vector_store=row["vector_store"],
        collection_name=row["collection_name"],
        system_profile=_build_system_profile_from_row(row),
        is_active=bool(row["is_active"]) if row["is_active"] is not None else None,
    ), None


async def activate_cabinet_async(
    db,
    cabinet_uuid: str,
) -> tuple[
    bool,
    "Cabinet | None",
    "SystemProfileItem | None",
    str | None,
]:
    import aiomysql

    cabinet_columns = await _get_cabinets_columns_async(db)
    if "system_profile_id" not in cabinet_columns:
        raise RuntimeError("cabinets table missing system_profile_id")
    if "is_active" not in cabinet_columns:
        raise RuntimeError("cabinets table missing is_active")

    cabinet_qry = """
    SELECT id, system_profile_id
    FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    update_qry = """
    UPDATE cabinets
    SET is_active = 1
    WHERE cabinet_uuid = %(cabinet_uuid)s
    """
    params = {"cabinet_uuid": cabinet_uuid}

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(cabinet_qry, params)
        cabinet_row = await cursor.fetchone()
        if cabinet_row is None:
            return False, None, None, "not_found"

        system_profile_id = cabinet_row.get("system_profile_id")
        if system_profile_id is None:
            return True, None, None, "system_profile_missing"

        system_profiles = await fetch_system_profiles_async(
            db, system_profile_id=int(system_profile_id)
        )
        if not system_profiles:
            return True, None, None, "system_profile_missing"

        system_profile = system_profiles[0]
        if not system_profile.is_active:
            return True, None, system_profile, "system_profile_inactive"

        await cursor.execute(update_qry, params)

    cabinet = await fetch_cabinet_by_uuid_async(db, cabinet_uuid=cabinet_uuid)
    if cabinet is None:
        return True, None, None, "update_failed"
    return True, cabinet, None, None


async def deactivate_cabinet_async(
    db,
    cabinet_uuid: str,
) -> tuple[bool, "Cabinet | None", str | None]:
    import aiomysql

    cabinet_columns = await _get_cabinets_columns_async(db)
    if "is_active" not in cabinet_columns:
        raise RuntimeError("cabinets table missing is_active")

    cabinet_qry = """
    SELECT id
    FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    update_qry = """
    UPDATE cabinets
    SET is_active = 0
    WHERE cabinet_uuid = %(cabinet_uuid)s
    """
    params = {"cabinet_uuid": cabinet_uuid}

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(cabinet_qry, params)
        cabinet_row = await cursor.fetchone()
        if cabinet_row is None:
            return False, None, "not_found"
        await cursor.execute(update_qry, params)

    cabinet = await fetch_cabinet_by_uuid_async(db, cabinet_uuid=cabinet_uuid)
    if cabinet is None:
        return True, None, "update_failed"
    return True, cabinet, None


async def delete_cabinet_async(
    db,
    cabinet_uuid: str,
) -> tuple[bool, str | None]:
    import aiomysql

    exists_qry = """
    SELECT id
    FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    documents_qry = """
    SELECT COUNT(*) AS document_count
    FROM documents
    WHERE cabinet_uuid = %(cabinet_uuid)s
    """
    delete_qry = """
    DELETE FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    """
    params = {"cabinet_uuid": cabinet_uuid}

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, params)
        exists = await cursor.fetchone()
        if exists is None:
            return False, "not_found"

        await cursor.execute(documents_qry, params)
        documents_row = await cursor.fetchone() or {}
        if int(documents_row.get("document_count") or 0) > 0:
            return False, "has_documents"

        await cursor.execute(delete_qry, params)
    return True, None


async def create_cabinet_async(
    db,
    payload: "CabinetCreateRequest",
) -> tuple[bool, "Cabinet | None"]:
    import aiomysql
    from uuid import uuid4

    from src.model.kl_models import Cabinet, CabinetCreateRequest

    if not isinstance(payload, CabinetCreateRequest):
        raise TypeError("payload must be CabinetCreateRequest")

    columns = await _get_cabinets_columns_async(db)
    column_names = set(columns.keys())
    storage_cols = await _resolve_cabinet_storage_columns_async(db)
    storage_type_col = storage_cols["storage_type"]
    base_path_col = storage_cols["storage_base_path"]
    upload_path_col = storage_cols["storage_path"]

    if "system_profile_id" not in column_names:
        raise RuntimeError("cabinets table missing system_profile_id")

    exists_qry = "SELECT id FROM system_profiles WHERE id = %(id)s LIMIT 1"
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, {"id": payload.system_profile_id})
        row = await cursor.fetchone()
    if row is None:
        return False, None

    insert_columns_sql = [
        "project_id",
        "cabinet_uuid",
        "cabinet_name",
        storage_type_col,
        base_path_col,
        upload_path_col,
        "vector_store",
        "collection_name",
        "system_profile_id",
    ]
    values_sql = ", ".join(
        [
            "%(project_id)s",
            "%(cabinet_uuid)s",
            "%(cabinet_name)s",
            "%(storage_type)s",
            "%(storage_base_path)s",
            "%(storage_path)s",
            "%(vector_store)s",
            "%(collection_name)s",
            "%(system_profile_id)s",
        ]
    )
    columns_sql = ", ".join(insert_columns_sql)
    insert_qry = f"INSERT INTO cabinets ({columns_sql}) VALUES ({values_sql})"
    select_sql = await _build_cabinet_select_sql(db, table_alias="c")
    select_qry = f"""
    SELECT {select_sql}
    FROM cabinets c
    LEFT JOIN system_profiles sp ON sp.id = c.system_profile_id
    LEFT JOIN llm_model_configs llmc ON llmc.id = sp.llm_model_config_id
    LEFT JOIN embedding_model_configs emc ON emc.id = sp.embedding_model_config_id
    LEFT JOIN models llmm ON llmm.id = llmc.model_id
    LEFT JOIN models emmm ON emmm.id = emc.model_id
    WHERE c.cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    cabinet_uuid = str(uuid4())
    params = {
        "project_id": payload.project_id,
        "cabinet_uuid": cabinet_uuid,
        "cabinet_name": payload.name,
        "vector_store": payload.vector_store,
        "collection_name": payload.collection_name,
        "storage_type": payload.storage_type,
        "storage_base_path": payload.storage_base_path,
        "storage_path": payload.storage_path,
        "system_profile_id": payload.system_profile_id,
    }

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(insert_qry, params)
        await cursor.execute(select_qry, {"cabinet_uuid": cabinet_uuid})
        row = await cursor.fetchone()

    if row is None:
        return True, None
    return True, Cabinet(
        id=row["id"],
        project_id=row["project_id"],
        cabinet_uuid=row["cabinet_uuid"],
        name=row["cabinet_name"],
        storage_type=row["storage_type"],
        storage_base_path=row["storage_base_path"],
        storage_path=row["storage_path"],
        vector_store=row["vector_store"],
        collection_name=row["collection_name"],
        system_profile=_build_system_profile_from_row(row),
        is_active=bool(row["is_active"]) if row["is_active"] is not None else None,
    )


def hash_password(password: str) -> str:
    from passlib.hash import argon2

    return argon2.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    from passlib.hash import argon2

    return argon2.verify(password, password_hash)


async def authenticate_user_async(
    db, email: str, password: str
) -> tuple[dict | None, str | None]:
    import aiomysql

    qry = """
    SELECT id, emp_id, username, email, password_hash, is_active
    FROM users
    WHERE email = %(email)s
    LIMIT 1
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, {"email": email})
        row = await cursor.fetchone()
    if row is None:
        return None, "not_found"
    password_hash = row.get("password_hash")
    if not password_hash or not verify_password(password, password_hash):
        return None, "invalid_password"
    if not row.get("is_active"):
        return None, "inactive"
    return row, None


async def create_user_async(
    db,
    payload: "UserCreateRequest",
) -> tuple[bool, "UserItem | None", str | None]:
    import aiomysql

    from src.model.kl_models import UserCreateRequest, UserItem

    if not isinstance(payload, UserCreateRequest):
        raise TypeError("payload must be UserCreateRequest")

    columns = await _get_table_columns_async(db, "users")
    check_fields: list[str] = []
    if "emp_id" in columns:
        check_fields.append("emp_id")
    if "email" in columns:
        check_fields.append("email")

    if check_fields:
        where_sql = " OR ".join(
            [f"{field} = %({field})s" for field in check_fields]
        )
        check_qry = f"""
        SELECT {", ".join(check_fields)}
        FROM users
        WHERE {where_sql}
        LIMIT 1
        """
        params = {
            "emp_id": payload.emp_id,
            "email": payload.email,
        }
        async with db.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(check_qry, params)
            row = await cursor.fetchone()
        if row is not None:
            matches: list[str] = []
            if "emp_id" in row and row.get("emp_id") == payload.emp_id:
                matches.append("emp_id")
            if "email" in row and row.get("email") == payload.email:
                matches.append("email")
            if matches:
                return False, None, ",".join(matches)
            return False, None, "user_exists"
    insert_fields: list[str] = []
    params: dict[str, object] = {}
    values = {
        "emp_id": payload.emp_id,
        "username": payload.username,
        "email": payload.email,
        "password_hash": hash_password(payload.password),
    }
    for key, value in values.items():
        if key in columns:
            insert_fields.append(key)
            params[key] = value
    if not insert_fields:
        raise RuntimeError("users has no writable columns")

    insert_columns_sql = ", ".join(insert_fields)
    insert_values_sql = ", ".join([f"%({key})s" for key in insert_fields])

    select_fields = [
        col
        for col in [
            "id",
            "emp_id",
            "username",
            "email",
            "is_active",
            "created_at",
        ]
        if col in columns
    ]
    if not select_fields:
        raise RuntimeError("users has no readable columns")
    select_sql = ", ".join(select_fields)

    insert_qry = f"""
    INSERT INTO users ({insert_columns_sql})
    VALUES ({insert_values_sql})
    """
    select_qry = f"""
    SELECT {select_sql}
    FROM users
    WHERE id = %(id)s
    LIMIT 1
    """

    try:
        async with db.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(insert_qry, params)
            user_id = cursor.lastrowid
            await cursor.execute(select_qry, {"id": user_id})
            row = await cursor.fetchone()
    except aiomysql.IntegrityError as exc:
        check_fields = []
        if "emp_id" in columns:
            check_fields.append("emp_id")
        if "email" in columns:
            check_fields.append("email")
        if check_fields:
            where_sql = " OR ".join(
                [f"{field} = %({field})s" for field in check_fields]
            )
            check_qry = f"""
            SELECT {", ".join(check_fields)}
            FROM users
            WHERE {where_sql}
            LIMIT 1
            """
            params = {
                "emp_id": payload.emp_id,
                "email": payload.email,
            }
            async with db.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(check_qry, params)
                row = await cursor.fetchone()
            if row is not None:
                matches: list[str] = []
                if "emp_id" in row and row.get("emp_id") == payload.emp_id:
                    matches.append("emp_id")
                if "email" in row and row.get("email") == payload.email:
                    matches.append("email")
                if matches:
                    return False, None, ",".join(matches)
        return False, None, str(exc) or "integrity_error"
    except Exception as exc:  # pragma: no cover - safety net
        return False, None, str(exc)

    if row is None:
        return False, None, "Failed to load created user"
    return True, UserItem(
        id=row.get("id"),
        emp_id=row.get("emp_id"),
        username=row.get("username"),
        email=row.get("email"),
        is_active=(
            bool(row.get("is_active"))
            if row.get("is_active") is not None
            else None
        ),
        created_at=row.get("created_at"),
    ), None


async def create_user_reset_token_async(
    db,
    payload: "UserResetRequest",
    ip_address: str | None = None,
) -> tuple[bool, dict[str, object] | None, str | None]:
    import aiomysql
    from datetime import datetime, timedelta
    from email.message import EmailMessage
    from secrets import token_urlsafe
    from smtplib import SMTP
    from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

    from src.config.settings import settings
    from src.model.kl_models import UserResetRequest

    if not isinstance(payload, UserResetRequest):
        raise TypeError("payload must be UserResetRequest")

    users_columns = await _get_table_columns_async(db, "users")
    reset_columns = await _get_table_columns_async(db, "user_reset_tokens")

    if "emp_id" not in users_columns or "email" not in users_columns:
        raise RuntimeError("users table missing emp_id or email")

    user_qry = """
    SELECT emp_id, email
    FROM users
    WHERE emp_id = %(emp_id)s
    LIMIT 1
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(user_qry, {"emp_id": payload.emp_id})
        user_row = await cursor.fetchone()
    if user_row is None or user_row.get("email") != payload.email:
        return False, None, "user_not_found"

    token = token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(minutes=10)

    insert_fields: list[str] = []
    params: dict[str, object] = {}
    values = {
        "emp_id": payload.emp_id,
        "email": payload.email,
        "token": token,
        "expires_at": expires_at,
        "used": 0,
        "ip_address": ip_address,
    }
    for key, value in values.items():
        if key in reset_columns and value is not None:
            insert_fields.append(key)
            params[key] = value

    if not insert_fields:
        raise RuntimeError("user_reset_tokens has no writable columns")

    insert_columns_sql = ", ".join(insert_fields)
    insert_values_sql = ", ".join([f"%({key})s" for key in insert_fields])

    insert_qry = f"""
    INSERT INTO user_reset_tokens ({insert_columns_sql})
    VALUES ({insert_values_sql})
    """

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(insert_qry, params)

    if not settings.reset_base_url:
        return False, None, "reset_base_url is not set"
    if not settings.smtp_user or not settings.smtp_pass:
        return False, None, "smtp credentials are not set"
    smtp_from = settings.smtp_from or settings.smtp_user

    parsed = urlparse(settings.reset_base_url)
    query = dict(parse_qsl(parsed.query))
    query["token"] = token
    reset_link = urlunparse(
        parsed._replace(query=urlencode(query, doseq=True))
    )

    message = EmailMessage()
    message["Subject"] = "Password reset"
    message["From"] = smtp_from
    message["To"] = payload.email
    message.set_content(
        "\n".join(
            [
                "비밀번호 초기화 요청을 받았습니다.",
                f"아래 링크에서 비밀번호를 재설정하세요 (10분 유효):",
                reset_link,
            ]
        )
    )

    try:
        with SMTP(settings.smtp_host, settings.smtp_port) as smtp:
            if settings.smtp_use_tls:
                smtp.starttls()
            smtp.login(settings.smtp_user, settings.smtp_pass)
            smtp.send_message(message)
    except Exception as exc:  # pragma: no cover - depends on SMTP
        return False, None, str(exc)

    return True, {"token": token, "expires_at": expires_at}, None


async def confirm_user_reset_token_async(
    db,
    payload: "UserResetConfirmRequest",
) -> tuple[bool, str | None]:
    import aiomysql
    from datetime import datetime

    from src.model.kl_models import UserResetConfirmRequest

    if not isinstance(payload, UserResetConfirmRequest):
        raise TypeError("payload must be UserResetConfirmRequest")

    reset_columns = await _get_table_columns_async(db, "user_reset_tokens")
    users_columns = await _get_table_columns_async(db, "users")

    if "token" not in reset_columns:
        raise RuntimeError("user_reset_tokens missing token column")
    if "password_hash" not in users_columns:
        raise RuntimeError("users missing password_hash column")

    select_fields = ["token"]
    for name in ("emp_id", "email", "expires_at", "used"):
        if name in reset_columns:
            select_fields.append(name)

    select_qry = f"""
    SELECT {", ".join(select_fields)}
    FROM user_reset_tokens
    WHERE token = %(token)s
    LIMIT 1
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(select_qry, {"token": payload.token})
        token_row = await cursor.fetchone()

    if token_row is None:
        return False, "invalid_token"

    expires_at = token_row.get("expires_at")
    if expires_at is not None and expires_at < datetime.utcnow():
        return False, "token_expired"

    used_value = token_row.get("used")
    if used_value is not None and int(used_value) != 0:
        return False, "token_used"

    criteria: list[str] = []
    params: dict[str, object] = {
        "password_hash": hash_password(payload.password),
    }
    if "emp_id" in users_columns and token_row.get("emp_id") is not None:
        criteria.append("emp_id = %(emp_id)s")
        params["emp_id"] = token_row.get("emp_id")
    if "email" in users_columns and token_row.get("email") is not None:
        criteria.append("email = %(email)s")
        params["email"] = token_row.get("email")

    if not criteria:
        return False, "token_missing_user"

    update_user_qry = f"""
    UPDATE users
    SET password_hash = %(password_hash)s
    WHERE {" AND ".join(criteria)}
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(update_user_qry, params)
        if cursor.rowcount == 0:
            return False, "user_not_found"

    update_fields: list[str] = []
    update_params: dict[str, object] = {"token": payload.token}
    now = datetime.utcnow()
    if "used" in reset_columns:
        update_fields.append("used = 1")
    if "used_at" in reset_columns:
        update_fields.append("used_at = %(now)s")
        update_params["now"] = now
    if "updated_at" in reset_columns:
        update_fields.append("updated_at = %(now)s")
        update_params["now"] = now

    if update_fields:
        update_reset_qry = f"""
        UPDATE user_reset_tokens
        SET {", ".join(update_fields)}
        WHERE token = %(token)s
        """
        async with db.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(update_reset_qry, update_params)

    return True, None


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


def _normalize_chunks_sort(sort: str | None) -> str:
    sort_map = {
        "created_at_desc": "ch.created_at DESC, ch.id DESC",
        "created_at_asc": "ch.created_at ASC, ch.id ASC",
        "index_asc": "ch.chunk_index ASC, ch.id ASC",
        "index_desc": "ch.chunk_index DESC, ch.id DESC",
        "doc_name_asc": "d.file_name ASC, ch.id ASC",
        "doc_name_desc": "d.file_name DESC, ch.id DESC",
        "chunk_id_asc": "ch.id ASC",
        "chunk_id_desc": "ch.id DESC",
    }
    return sort_map.get(sort or "", "ch.created_at DESC, ch.id DESC")


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

    if not cabinet.storage_base_path:
        raise RuntimeError("Cabinet storage_base_path is missing")
    storage_type = (cabinet.storage_type or "").upper()
    if storage_type and storage_type not in ("LOCAL", "FILESYSTEM", "NFS"):
        raise RuntimeError(f"Unsupported storage_type: {cabinet.storage_type}")

    def _is_writable_dir(path: Path) -> bool:
        if path.exists():
            return path.is_dir() and os.access(path, os.W_OK)
        parent = path.parent
        return parent.exists() and os.access(parent, os.W_OK)

    base_dir = Path(cabinet.storage_base_path)
    if base_dir.is_absolute() and not _is_writable_dir(base_dir):
        fallback_dir = (Path.cwd() / "upload").resolve()
        if _is_writable_dir(fallback_dir):
            base_dir = fallback_dir
        else:
            raise RuntimeError(
                f"storage_base_path is not writable: {cabinet.storage_base_path}"
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
                f"storage_base_path is not writable: {base_dir}"
            )
    logger.info(
        "upload target resolved: root=%s storage_path=%s final=%s",
        cabinet.storage_base_path,
        cabinet.storage_path,
        base_dir,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    columns = await _get_documents_columns_async(db)
    if "cabinet_uuid" not in columns:
        raise RuntimeError("documents table is missing cabinet_uuid")
    if "chunking_run_id" in columns:
        latest_run_qry = """
        SELECT cr.id
        FROM cabinets c
        JOIN chunking_runs cr ON cr.cabinet_uuid = c.cabinet_uuid
        WHERE c.cabinet_uuid = %(cabinet_uuid)s
        ORDER BY cr.created_at DESC, cr.id DESC
        LIMIT 1
        """
        async with db.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(
                latest_run_qry, {"cabinet_uuid": cabinet.cabinet_uuid}
            )
            run_row = await cursor.fetchone()
        if run_row is None:
            raise RuntimeError(
                "chunking_run_id is required but no chunking run exists for cabinet"
            )
        latest_chunking_run_id = int(run_row["id"])
    else:
        latest_chunking_run_id = None

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
            doc_uuid = str(uuid4())
            saved_name = f"{doc_uuid}{suffix}"
            target_path = base_dir / saved_name
            if target_path.exists():
                doc_uuid = str(uuid4())
                saved_name = f"{doc_uuid}{suffix}"
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
                "doc_uuid": doc_uuid,
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
            values["cabinet_uuid"] = cabinet.cabinet_uuid
            if latest_chunking_run_id is not None:
                values["chunking_run_id"] = latest_chunking_run_id

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
                elif name == "doc_uuid":
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
                            "documents.doc_uuid must be AUTO_INCREMENT or have a default"
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
            status_value = params.get("status") or (
                columns.get("status", {}).get("column_default") or "UPLOADED"
            )
            step_value = params.get("processing_step") or (
                columns.get("processing_step", {}).get("column_default") or "PENDING"
            )

            items.append(
                DocumentListItem(
                    document_uuid=str(doc_uuid),
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
        d.doc_uuid AS document_uuid,
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
            document_uuid=row["document_uuid"],
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


async def fetch_document_download_info_async(
    db,
    document_uuid: str,
) -> dict[str, str] | None:
    import aiomysql
    from pathlib import Path

    columns = await _get_documents_columns_async(db)
    path_column = None
    for candidate in ("storage_path", "file_path", "path"):
        if candidate in columns:
            path_column = candidate
            break

    cabinets_columns = await _get_cabinets_columns_async(db)
    if "cabinet_uuid" not in cabinets_columns:
        raise RuntimeError("cabinets table missing uuid column")
    if "cabinet_uuid" not in columns:
        raise RuntimeError("documents table missing cabinet_uuid column")

    storage_cols = await _resolve_cabinet_storage_columns_async(db)
    base_path_col = storage_cols["storage_base_path"]
    storage_path_col = storage_cols["storage_path"]

    select_path_sql = f", d.{path_column} AS file_path" if path_column else ""
    qry = f"""
    SELECT
        d.doc_uuid AS document_uuid,
        d.file_name,
        d.file_type,
        c.{base_path_col} AS storage_base_path,
        c.{storage_path_col} AS storage_path
        {select_path_sql}
    FROM documents d
    JOIN cabinets c ON d.cabinet_uuid = c.cabinet_uuid
    WHERE d.doc_uuid = %(document_uuid)s
    LIMIT 1
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, {"document_uuid": document_uuid})
        row = await cursor.fetchone()
    if row is None:
        return None
    file_name = row.get("file_name")
    file_type = row.get("file_type")
    if not file_name or not file_type:
        return None

    file_path = row.get("file_path")
    if file_path:
        return {"file_path": str(file_path), "file_name": str(file_name)}

    storage_base_path = row.get("storage_base_path")
    storage_path = row.get("storage_path")
    if not storage_base_path:
        return None
    safe_type = str(file_type).lstrip(".").strip() or "bin"
    saved_name = f"{row.get('document_uuid')}.{safe_type}"

    base_dir = Path(str(storage_base_path))
    fallback_dir = (Path.cwd() / "upload").resolve()
    if storage_path:
        storage_path = str(storage_path)
        if Path(storage_path).is_absolute():
            storage_path = storage_path.lstrip("/")
        base_dir = base_dir / storage_path
    resolved_path = base_dir / saved_name
    if resolved_path.exists():
        return {"file_path": str(resolved_path), "file_name": str(file_name)}
    fallback_path = fallback_dir / saved_name
    return {
        "file_path": str(fallback_path if fallback_path.exists() else resolved_path),
        "file_name": str(file_name),
    }


async def delete_document_async(
    db,
    document_uuid: str,
) -> tuple[bool, str | None]:
    import aiomysql
    import os

    info = await fetch_document_download_info_async(
        db, document_uuid=document_uuid
    )
    exists_qry = """
    SELECT doc_uuid, processing_step
    FROM documents
    WHERE doc_uuid = %(document_uuid)s
    LIMIT 1
    """
    delete_qry = "DELETE FROM documents WHERE doc_uuid = %(document_uuid)s"
    params = {"document_uuid": document_uuid}

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(exists_qry, params)
        row = await cursor.fetchone()
        if row is None:
            return False, "not_found"
        if str(row.get("processing_step") or "").upper() == "PARSING":
            return False, "parsing"

    file_path = info["file_path"] if info else None
    if file_path:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            return False, "file_delete_failed"

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(delete_qry, params)
    return True, None


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


async def fetch_chunks_async(
    db,
    cabinet_uuid: str | None,
    page: int,
    page_size: int,
    preview_length: int,
    document_uuid: str | None,
    chunking_run_id: int | None,
    sort: str | None,
) -> tuple[list["ChunkListItem"], int]:
    import aiomysql

    from src.model.kl_models import ChunkListItem

    chunking_configs_columns = await _get_table_columns_async(db, "chunking_configs")
    splitter_version_select = (
        "cc.splitter_version AS splitter_version"
        if "splitter_version" in chunking_configs_columns
        else "NULL AS splitter_version"
    )
    join_sql = ""
    where_clauses = ["1=1"]
    params: dict[str, object] = {"preview_length": preview_length}
    if cabinet_uuid:
        join_sql, base_where_sql = await _resolve_documents_cabinet_filter_async(db)
        base_clause = base_where_sql.replace("WHERE ", "", 1)
        where_clauses = [base_clause]
        params["cabinet_uuid"] = cabinet_uuid
    if document_uuid:
        where_clauses.append("d.doc_uuid = %(document_uuid)s")
        params["document_uuid"] = document_uuid
    if chunking_run_id is not None:
        where_clauses.append("ch.chunking_run_id = %(chunking_run_id)s")
        params["chunking_run_id"] = chunking_run_id

    where_sql = " AND ".join(where_clauses)
    sort_sql = _normalize_chunks_sort(sort)
    offset = (page - 1) * page_size
    params.update({"offset": offset, "limit": page_size})

    count_qry = f"""
    SELECT COUNT(*) AS total
    FROM chunks ch
    JOIN documents d ON d.doc_uuid = ch.doc_uuid
    {join_sql}
    JOIN chunking_runs cr ON cr.id = ch.chunking_run_id
    WHERE {where_sql}
    """
    list_qry = f"""
    SELECT
        ch.id,
        ch.doc_uuid,
        ch.chunking_run_id,
        ch.chunk_index,
        LEFT(ch.content, %(preview_length)s) AS content_preview,
        ch.content,
        ch.created_at,
        ch.updated_at,
        d.file_name,
        d.file_type,
        cr.chunking_config_id,
        cr.chunk_size,
        cr.chunk_overlap,
        cr.unit,
        {splitter_version_select},
        cc.method_name
    FROM chunks ch
    JOIN documents d ON d.doc_uuid = ch.doc_uuid
    {join_sql}
    JOIN chunking_runs cr ON cr.id = ch.chunking_run_id
    LEFT JOIN chunking_configs cc ON cc.id = cr.chunking_config_id
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
        ChunkListItem(
            id=row["id"],
            doc_uuid=row["doc_uuid"],
            document_name=row["file_name"],
            document_file_type=row["file_type"],
            chunking_run_id=row["chunking_run_id"],
            chunk_index=row["chunk_index"],
            content_preview=row["content_preview"],
            content=row["content"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            chunking_config_id=row["chunking_config_id"],
            method_name=row["method_name"],
            chunk_size=row["chunk_size"],
            chunk_overlap=row["chunk_overlap"],
            unit=row["unit"],
            splitter_version=row["splitter_version"],
        )
        for row in rows
    ]
    return items, total_items


async def fetch_cabinet_chunk_stats_async(
    db,
    cabinet_uuid: str,
) -> tuple[bool, int, float]:
    import aiomysql

    cabinet_qry = """
    SELECT id
    FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    stats_qry = """
    SELECT
        COUNT(ch.id) AS total_chunks,
        COALESCE(ROUND(AVG(CHAR_LENGTH(ch.content)), 0), 0) AS avg_chunk_length
    FROM documents d
    LEFT JOIN chunks ch ON ch.doc_uuid = d.doc_uuid
    WHERE d.cabinet_uuid = %(cabinet_uuid)s
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(cabinet_qry, {"cabinet_uuid": cabinet_uuid})
        cabinet_row = await cursor.fetchone()
        if cabinet_row is None:
            return False, 0, 0.0
        await cursor.execute(stats_qry, {"cabinet_uuid": cabinet_uuid})
        stats_row = await cursor.fetchone()

    if not stats_row:
        return True, 0, 0.0
    total_chunks = int(stats_row["total_chunks"] or 0)
    avg_chunk_length = float(stats_row["avg_chunk_length"] or 0)
    return True, total_chunks, avg_chunk_length


async def fetch_cabinet_qa_summary_async(
    db,
    cabinet_uuid: str,
) -> tuple[bool, int, int, int, float, int]:
    import aiomysql

    cabinet_qry = """
    SELECT id
    FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    summary_qry = """
    SELECT
        COUNT(DISTINCT d.doc_uuid) AS total_documents,
        COUNT(DISTINCT qa.id) AS total_qa,
        COUNT(DISTINCT qe.qa_id) AS evaluated_qa,
        COALESCE(AVG(qe.score), 0) AS avg_score
    FROM documents d
    JOIN chunks ch ON ch.doc_uuid = d.doc_uuid
    JOIN qa ON qa.chunk_id = ch.id
    LEFT JOIN qa_evaluations qe ON qe.qa_id = qa.id
    WHERE d.cabinet_uuid = %(cabinet_uuid)s
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(cabinet_qry, {"cabinet_uuid": cabinet_uuid})
        cabinet_row = await cursor.fetchone()
        if cabinet_row is None:
            return False, 0, 0, 0, 0.0, 0
        await cursor.execute(summary_qry, {"cabinet_uuid": cabinet_uuid})
        row = await cursor.fetchone()

    if not row:
        return True, 0, 0, 0, 0.0, 0
    total_documents = int(row["total_documents"] or 0)
    total_qa = int(row["total_qa"] or 0)
    evaluated_qa = int(row["evaluated_qa"] or 0)
    avg_score = float(row["avg_score"] or 0)
    unevaluated_qa = max(total_qa - evaluated_qa, 0)
    return True, total_documents, total_qa, evaluated_qa, avg_score, unevaluated_qa


async def create_qa_evaluation_async(
    db,
    qa_id: int,
    evaluator_type: str,
    score: int,
    feedback: str,
) -> tuple[bool, int | None]:
    import aiomysql

    evaluation_columns = await _get_table_columns_async(db, "qa_evaluations")
    required_columns = {"qa_id", "evaluator_type", "score", "feedback"}
    missing = required_columns - evaluation_columns
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise RuntimeError(
            f"qa_evaluations table missing columns: {missing_list}"
        )

    qa_qry = """
    SELECT id
    FROM qa
    WHERE id = %(qa_id)s
    LIMIT 1
    """
    insert_qry = """
    INSERT INTO qa_evaluations (
        qa_id,
        evaluator_type,
        score,
        feedback
    ) VALUES (
        %(qa_id)s,
        %(evaluator_type)s,
        %(score)s,
        %(feedback)s
    )
    """
    params = {
        "qa_id": qa_id,
        "evaluator_type": evaluator_type,
        "score": score,
        "feedback": feedback,
    }

    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qa_qry, {"qa_id": qa_id})
        qa_row = await cursor.fetchone()
        if qa_row is None:
            return False, None
        await cursor.execute(insert_qry, params)
        evaluation_id = cursor.lastrowid

    return True, int(evaluation_id) if evaluation_id is not None else None


async def fetch_cabinet_qa_list_async(
    db,
    cabinet_uuid: str,
    document_uuid: str | None,
    page: int,
    page_size: int,
) -> tuple[bool, list["QAListItem"], int]:
    import aiomysql

    from src.model.kl_models import QAListItem, QAEvaluationItem

    cabinet_qry = """
    SELECT id
    FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    count_qry = """
    SELECT COUNT(DISTINCT qa.id) AS total
    FROM documents d
    JOIN chunks ch ON ch.doc_uuid = d.doc_uuid
    JOIN qa ON qa.chunk_id = ch.id
    WHERE d.cabinet_uuid = %(cabinet_uuid)s
      AND (%(document_uuid)s IS NULL OR d.doc_uuid = %(document_uuid)s)
    """
    list_qry = """
    SELECT
        qa.id AS id,
        qa.chunk_id AS chunk_id,
        ch.chunk_index AS chunk_index,
        d.doc_uuid AS document_uuid,
        d.file_name AS document_name,
        qa.question AS question,
        qa.answer AS answer,
        qa.generated_by AS generated_by,
        qa.created_at AS created_at,
        qa.updated_at AS updated_at
    FROM documents d
    JOIN chunks ch ON ch.doc_uuid = d.doc_uuid
    JOIN qa ON qa.chunk_id = ch.id
    WHERE d.cabinet_uuid = %(cabinet_uuid)s
      AND (%(document_uuid)s IS NULL OR d.doc_uuid = %(document_uuid)s)
    ORDER BY d.created_at DESC, qa.created_at DESC, qa.id DESC
    LIMIT %(offset)s, %(limit)s
    """
    evaluations_qry = """
    SELECT
        qe.qa_id AS qa_id,
        qe.evaluator_type AS evaluator_type,
        qe.score AS score,
        qe.feedback AS feedback,
        qe.created_at AS evaluated_at
    FROM qa_evaluations qe
    WHERE qe.qa_id IN ({qa_ids})
    ORDER BY qe.qa_id DESC, qe.created_at DESC
    """
    offset = (page - 1) * page_size
    params = {
        "cabinet_uuid": cabinet_uuid,
        "document_uuid": document_uuid,
        "offset": offset,
        "limit": page_size,
    }
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(cabinet_qry, {"cabinet_uuid": cabinet_uuid})
        cabinet_row = await cursor.fetchone()
        if cabinet_row is None:
            return False, [], 0
        await cursor.execute(count_qry, params)
        count_row = await cursor.fetchone()
        total_items = int(count_row["total"]) if count_row else 0
        await cursor.execute(list_qry, params)
        rows = await cursor.fetchall()

    items_by_id: dict[int, QAListItem] = {}
    ordered_ids: list[int] = []
    for row in rows:
        qa_id = int(row["id"])
        item = QAListItem(
            id=qa_id,
            chunk_id=int(row["chunk_id"]),
            chunk_index=row["chunk_index"],
            document_uuid=row["document_uuid"],
            document_name=row["document_name"],
            question=row["question"],
            answer=row["answer"],
            generated_by=row["generated_by"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            evaluation_count=0,
            avg_score=None,
            evaluations=[],
        )
        items_by_id[qa_id] = item
        ordered_ids.append(qa_id)

    if ordered_ids:
        qa_id_placeholders = ", ".join(
            [f"%({index})s" for index in range(len(ordered_ids))]
        )
        qa_id_params = {
            str(index): qa_id for index, qa_id in enumerate(ordered_ids)
        }
        async with db.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(
                evaluations_qry.format(qa_ids=qa_id_placeholders),
                qa_id_params,
            )
            evaluation_rows = await cursor.fetchall()
        for row in evaluation_rows:
            qa_id = int(row["qa_id"])
            item = items_by_id.get(qa_id)
            if item is None:
                continue
            item.evaluations.append(
                QAEvaluationItem(
                    evaluator_type=row["evaluator_type"],
                    score=row["score"],
                    feedback=row["feedback"],
                    evaluated_at=row["evaluated_at"],
                )
            )

    for item in items_by_id.values():
        scores = [e.score for e in item.evaluations if e.score is not None]
        item.evaluation_count = len(scores)
        if scores:
            item.avg_score = round(sum(scores) / len(scores), 1)
        else:
            item.avg_score = None

    items = [items_by_id[qa_id] for qa_id in ordered_ids]
    return True, items, total_items


async def fetch_cabinet_document_summary_async(
    db,
    cabinet_uuid: str,
) -> tuple[bool, int, int, list["DocumentQASummaryItem"]]:
    import aiomysql

    from src.model.kl_models import DocumentQASummaryItem

    cabinet_qry = """
    SELECT id
    FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    summary_qry = """
    SELECT
        COUNT(DISTINCT d.doc_uuid) AS total_documents,
        COUNT(DISTINCT qa.id) AS total_qa
    FROM documents d
    LEFT JOIN chunks ch ON ch.doc_uuid = d.doc_uuid
    LEFT JOIN qa ON qa.chunk_id = ch.id
    WHERE d.cabinet_uuid = %(cabinet_uuid)s
    """
    list_qry = """
    SELECT
        d.doc_uuid AS document_uuid,
        d.file_name AS file_name,
        d.created_at AS created_at,
        d.file_size AS file_size,
        COUNT(DISTINCT ch.id) AS chunks,
        COUNT(DISTINCT qa.id) AS qa,
        SUM(
            CASE
                WHEN qe.eval_count IS NULL OR qe.eval_count = 0 THEN 1
                ELSE 0
            END
        ) AS unevaluated_qa
    FROM documents d
    LEFT JOIN chunks ch ON ch.doc_uuid = d.doc_uuid
    LEFT JOIN qa ON qa.chunk_id = ch.id
    LEFT JOIN (
        SELECT qa_id, COUNT(*) AS eval_count
        FROM qa_evaluations
        GROUP BY qa_id
    ) qe ON qe.qa_id = qa.id
    WHERE d.cabinet_uuid = %(cabinet_uuid)s
    GROUP BY d.doc_uuid, d.file_name, d.created_at, d.file_size
    ORDER BY d.created_at DESC, d.doc_uuid DESC
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(cabinet_qry, {"cabinet_uuid": cabinet_uuid})
        cabinet_row = await cursor.fetchone()
        if cabinet_row is None:
            return False, 0, 0, []
        await cursor.execute(summary_qry, {"cabinet_uuid": cabinet_uuid})
        summary_row = await cursor.fetchone()
        await cursor.execute(list_qry, {"cabinet_uuid": cabinet_uuid})
        rows = await cursor.fetchall()

    summary_row = summary_row or {}
    total_documents = int(summary_row.get("total_documents") or 0)
    total_qa = int(summary_row.get("total_qa") or 0)
    items = [
        DocumentQASummaryItem(
            file_name=row["file_name"],
            document_uuid=row["document_uuid"],
            created_at=row["created_at"],
            file_size=row["file_size"],
            chunks=int(row["chunks"] or 0),
            qa=int(row["qa"] or 0),
            unevaluated_qa=int(row["unevaluated_qa"] or 0),
        )
        for row in rows
    ]
    return True, total_documents, total_qa, items


async def fetch_enquery_summary_async(db) -> dict[str, object]:
    import aiomysql

    qry = """
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN action = 'WAITING' THEN 1 ELSE 0 END) AS waiting_count,
        SUM(CASE WHEN action = 'APPROVE' THEN 1 ELSE 0 END) AS approve_count,
        SUM(CASE WHEN action = 'EDIT' THEN 1 ELSE 0 END) AS edit_count,
        SUM(CASE WHEN action = 'REJECT' THEN 1 ELSE 0 END) AS reject_count,
        AVG(score) AS avg_score
    FROM enquery_answers
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry)
        row = await cursor.fetchone()

    row = row or {}
    return {
        "total": int(row.get("total") or 0),
        "waiting": int(row.get("waiting_count") or 0),
        "approve": int(row.get("approve_count") or 0),
        "edit": int(row.get("edit_count") or 0),
        "reject": int(row.get("reject_count") or 0),
        "avg_score": (
            float(row["avg_score"]) if row.get("avg_score") is not None else None
        ),
    }


async def fetch_enqueries_async(
    db,
    page: int,
    page_size: int,
) -> tuple[list["EnqueryItem"], int]:
    import aiomysql

    from src.model.kl_models import EnqueryCitedChunk, EnqueryItem

    total_qry = "SELECT COUNT(*) AS total FROM enquery_answers"
    list_qry = """
    SELECT
        id,
        task_id,
        channel,
        question,
        ai_answer,
        human_answer,
        final_answer,
        action,
        human,
        score,
        reason,
        created_at,
        updated_at
    FROM enquery_answers
    ORDER BY created_at DESC, id DESC
    LIMIT %(limit)s OFFSET %(offset)s
    """
    offset = (page - 1) * page_size
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(total_qry)
        total_row = await cursor.fetchone()
        await cursor.execute(list_qry, {"limit": page_size, "offset": offset})
        rows = await cursor.fetchall()

    total_items = int((total_row or {}).get("total") or 0)
    if not rows:
        return [], total_items

    enquery_ids = [row["id"] for row in rows]
    if not enquery_ids:
        return [], total_items
    placeholders = ", ".join(["%s"] * len(enquery_ids))
    cited_qry = f"""
    SELECT
        cd.enquery_task_id AS enquery_id,
        c.id AS chunk_id,
        c.doc_uuid AS doc_uuid,
        c.chunk_index AS chunk_index,
        c.content AS content,
        d.file_name AS file_name,
        d.file_type AS file_type,
        d.file_size AS file_size
    FROM cited_documents cd
    JOIN chunks c ON c.id = cd.chunk_id
    JOIN documents d ON d.doc_uuid = c.doc_uuid
    WHERE cd.enquery_task_id IN ({placeholders})
    ORDER BY cd.enquery_task_id ASC, c.chunk_index ASC, c.id ASC
    """
    cited_map: dict[int, list[EnqueryCitedChunk]] = {}
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(cited_qry, enquery_ids)
        cited_rows = await cursor.fetchall()
    for row in cited_rows:
        enquery_id = int(row["enquery_id"])
        cited_map.setdefault(enquery_id, []).append(
            EnqueryCitedChunk(
                chunk_id=row["chunk_id"],
                doc_uuid=row["doc_uuid"],
                chunk_index=row.get("chunk_index"),
                content=row.get("content"),
                file_name=row.get("file_name"),
                file_type=row.get("file_type"),
                file_size=row.get("file_size"),
            )
        )

    items: list[EnqueryItem] = []
    for row in rows:
        items.append(
            EnqueryItem(
                id=row["id"],
                task_id=row["task_id"],
                channel=row["channel"],
                question=row["question"],
                ai_answer=row.get("ai_answer"),
                human_answer=row.get("human_answer"),
                final_answer=row.get("final_answer"),
                action=row.get("action"),
                human=row.get("human"),
                score=row.get("score"),
                reason=row.get("reason"),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
                cited_chunks=cited_map.get(int(row["id"]), []),
            )
        )
    return items, total_items


async def enqueue_document_pipeline_async(
    redis_client,
    cabinet_uuid: str,
    items: list["DocumentListItem"],
) -> int:
    if not items:
        return 0

    enqueued = 0
    for item in items:
        fields = {
            "doc_uuid": item.document_uuid,
            "cabinet_uuid": cabinet_uuid,
            "file_name": item.file_name or "",
            "file_type": item.file_type or "",
            "file_size": str(item.file_size or 0),
            "uploaded_at": (
                item.uploaded_at.isoformat()
                if item.uploaded_at is not None
                else ""
            ),
        }
        await redis_client.xadd(
            redis_client.stream,
            fields,
            maxlen=redis_client.stream_maxlen,
        )
        enqueued += 1
    return enqueued


async def fetch_cabinet_chunking_settings_async(
    db,
    cabinet_uuid: str,
) -> tuple[bool, "ChunkingRun | None", list["ChunkingConfig"]]:
    import aiomysql

    from src.model.kl_models import ChunkingConfig, ChunkingRun

    chunking_configs_columns = await _get_table_columns_async(db, "chunking_configs")

    configs_columns = [
        "id",
        "method_name",
        "chunk_size",
        "chunk_overlap",
        "unit",
    ]
    if "memo" in chunking_configs_columns:
        configs_columns.append("memo")
    if "splitter_version" in chunking_configs_columns:
        configs_columns.append("splitter_version")

    runs_columns = [
        "cr.id AS id",
        "cr.chunking_config_id AS chunking_config_id",
        "cr.cabinet_uuid AS cabinet_uuid",
        "cr.chunk_size AS chunk_size",
        "cr.chunk_overlap AS chunk_overlap",
        "cr.unit AS unit",
        "cr.created_at AS created_at",
        "cr.updated_at AS updated_at",
    ]
    if "memo" in chunking_configs_columns:
        runs_columns.append("cc.memo AS memo")
    if "splitter_version" in chunking_configs_columns:
        runs_columns.append("cc.splitter_version AS splitter_version")

    cabinet_qry = """
    SELECT id
    FROM cabinets
    WHERE cabinet_uuid = %(cabinet_uuid)s
    LIMIT 1
    """
    configs_qry = f"""
    SELECT
        {", ".join(configs_columns)}
    FROM chunking_configs
    ORDER BY id
    """
    current_run_qry = f"""
    SELECT
        {", ".join(runs_columns)}
    FROM chunking_runs cr
    LEFT JOIN chunking_configs cc ON cc.id = cr.chunking_config_id
    WHERE cr.cabinet_uuid = %(cabinet_uuid)s
    ORDER BY cr.created_at DESC, cr.id DESC
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

    chunking_configs_columns = await _get_table_columns_async(db, "chunking_configs")
    runs_columns = [
        "cr.id AS id",
        "cr.chunking_config_id AS chunking_config_id",
        "cr.cabinet_uuid AS cabinet_uuid",
        "cr.chunk_size AS chunk_size",
        "cr.chunk_overlap AS chunk_overlap",
        "cr.unit AS unit",
        "cr.created_at AS created_at",
        "cr.updated_at AS updated_at",
    ]
    if "memo" in chunking_configs_columns:
        runs_columns.append("cc.memo AS memo")
    if "splitter_version" in chunking_configs_columns:
        runs_columns.append("cc.splitter_version AS splitter_version")

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
        unit
    ) VALUES (
        %(chunking_config_id)s,
        %(cabinet_uuid)s,
        %(chunk_size)s,
        %(chunk_overlap)s,
        %(unit)s
    )
    """
    select_qry = f"""
    SELECT
        {", ".join(runs_columns)}
    FROM chunking_runs cr
    LEFT JOIN chunking_configs cc ON cc.id = cr.chunking_config_id
    WHERE cr.cabinet_uuid = %(cabinet_uuid)s
    ORDER BY cr.created_at DESC, cr.id DESC
    LIMIT 1
    """

    params = {
        "chunking_config_id": chunking_run.chunking_config_id,
        "cabinet_uuid": cabinet_uuid,
        "chunk_size": chunking_run.chunk_size,
        "chunk_overlap": chunking_run.chunk_overlap,
        "unit": chunking_run.unit,
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
_TABLE_COLUMNS: dict[str, set[str]] | None = None


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


async def _get_table_columns_async(
    db,
    table_name: str,
) -> set[str]:
    import aiomysql

    global _TABLE_COLUMNS
    if _TABLE_COLUMNS is None:
        _TABLE_COLUMNS = {}
    if table_name in _TABLE_COLUMNS:
        return _TABLE_COLUMNS[table_name]

    qry = """
    SELECT COLUMN_NAME AS column_name
    FROM information_schema.columns
    WHERE table_schema = DATABASE()
      AND table_name = %(table_name)s
    """
    async with db.cursor(aiomysql.DictCursor) as cursor:
        await cursor.execute(qry, {"table_name": table_name})
        rows = await cursor.fetchall()

    columns = {row["column_name"] for row in rows}
    _TABLE_COLUMNS[table_name] = columns
    return columns


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
    else:
        raise RuntimeError("documents table is missing cabinet_uuid")

    return _DOCUMENTS_CABINET_FILTER
