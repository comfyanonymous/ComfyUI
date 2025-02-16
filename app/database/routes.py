from app.database.db import db
from aiohttp import web

def create_routes(
    routes, prefix, entity, get=False, get_by_id=False, post=False, delete=False
):
    if get:
        @routes.get(f"/{prefix}/{table}")
        async def get_table(request):
            connection = db.get_connection()
            cursor = connection.cursor()
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            return web.json_response(rows)
        
    if get_by_id:
        @routes.get(f"/{prefix}/{table}/{id}")
        async def get_table_by_id(request):
            connection = db.get_connection()
            cursor = connection.cursor()
            cursor.execute(f"SELECT * FROM {table} WHERE id = {id}")
            row = cursor.fetchone()
            return web.json_response(row)
        
    if post:
        @routes.post(f"/{prefix}/{table}")
        async def post_table(request):
            data = await request.json()
            connection = db.get_connection()
            cursor = connection.cursor()
            cursor.execute(f"INSERT INTO {table} ({data}) VALUES ({data})")
            return web.json_response({"status": "success"})
