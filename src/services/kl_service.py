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


def hash_password(password: str) -> str:
    from passlib.hash import argon2

    return argon2.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    from passlib.hash import argon2

    return argon2.verify(password, password_hash)


async def authenticate_user_async(db, email: str, password: str) -> dict | None:
    import aiomysql

    qry = """
    SELECT id, emp_id, email, password_hash
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
