import os
from notion_client import Client
from datetime import datetime

# 设置你的 Notion API Token 和 数据库 ID
NOTION_TOKEN = "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"
NOTION_DATABASE_ID = "230de91f217680388daed107820a5dc8"

if not NOTION_TOKEN or not NOTION_DATABASE_ID:
    print("请设置 NOTION_TOKEN 和 NOTION_DATABASE_ID 环境变量。")
    exit()

# 初始化 Notion 客户端
notion = Client(auth=NOTION_TOKEN)

def get_all_records_from_database():
    """
    从 Notion 数据库获取所有记录的 ID。
    """
    records_id = []
    has_more = True
    next_cursor = None

    print(f"正在从数据库 '{NOTION_DATABASE_ID}' 获取记录...")
    while has_more:
        try:
            # 查询数据库
            # 注意: Notion API 对查询结果有分页限制 (通常每页 100 条)
            response = notion.databases.query(
                database_id=NOTION_DATABASE_ID,
                start_cursor=next_cursor
            )

            for page in response["results"]:
                records_id.append(page["id"])

            has_more = response["has_more"]
            next_cursor = response["next_cursor"]
            
        except Exception as e:
            print(f"获取记录失败: {e}")
            break
            
    print(f"找到 {len(records_id)} 条记录。")
    return records_id

def archive_record(page_id):
    """
    归档（删除）指定的 Notion 页面（记录）。
    """
    try:
        notion.pages.update(
            page_id=page_id,
            archived=True
        )
        print(f"成功归档页面: {page_id}")
    except Exception as e:
        print(f"归档页面 {page_id} 失败: {e}")

def delete_all_records():
    """
    删除数据库中所有现有记录，然后插入新的记录。

    Args:
        new_record_data (list of dict): 包含要插入的新记录数据的列表，
                                         每个字典应包含 'name', 'tags', 'status' 等键。
    """
    print("\n--- 步骤 1: 删除（归档）所有现有记录 ---")
    record_ids_to_delete = get_all_records_from_database()

    if not record_ids_to_delete:
        print("数据库中没有需要删除的记录。")
    else:
        for page_id in record_ids_to_delete:
            archive_record(page_id)
        print("所有现有记录已归档。")

def add_record_to_notion_database(url):
    """
    向 Notion 数据库插入一条新记录。

    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    try:
        new_page = notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties={
                "更新时间": {
                    "title": [
                        {
                            "text": {"content": formatted_datetime}
                        }
                    ]
                },
                "地址": {
                    "url": url # <<--- 传递一个简单的 URL 字符串
                }
                # 根据你的数据库结构添加更多属性
            },
        )
        print(f"成功插入记录: {url}")
    except Exception as e:
        print(f"插入记录失败: {e}")
