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
