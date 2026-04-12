"""Async DB helpers that wrap sync SQLAlchemy with run_in_executor."""
import asyncio
from functools import partial
from research_api.db import create_session
from research_api.models import Project, Intent, PaperAsset, ClaimAsset, Source, FeedItem, StyleAsset
from app.database.models import to_dict


def _sync_list_projects():
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(Project).order_by(Project.updated_at.desc()))
        return [to_dict(p) for p in result.scalars().all()]


def _sync_create_project(data):
    with create_session() as session:
        project = Project(**data)
        session.add(project)
        session.commit()
        session.refresh(project)
        return to_dict(project)


def _sync_get_project(project_id):
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(Project).where(Project.id == project_id))
        p = result.scalar_one_or_none()
        return to_dict(p) if p else None


def _sync_list_intents(project_id):
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(
            select(Intent).where(Intent.project_id == project_id).order_by(Intent.priority.desc())
        )
        return [to_dict(i) for i in result.scalars().all()]


def _sync_create_intent(data):
    with create_session() as session:
        intent = Intent(**data)
        session.add(intent)
        session.commit()
        session.refresh(intent)
        return to_dict(intent)


def _sync_update_intent(intent_id, data):
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(Intent).where(Intent.id == intent_id))
        intent = result.scalar_one_or_none()
        if not intent:
            return None
        for key, value in data.items():
            if hasattr(intent, key):
                setattr(intent, key, value)
        session.commit()
        session.refresh(intent)
        return to_dict(intent)


def _sync_list_papers(library_status=None, read_status=None):
    with create_session() as session:
        from sqlalchemy import select
        query = select(PaperAsset)
        if library_status:
            query = query.where(PaperAsset.library_status == library_status)
        if read_status:
            query = query.where(PaperAsset.read_status == read_status)
        query = query.order_by(PaperAsset.updated_at.desc())
        result = session.execute(query)
        return [to_dict(p) for p in result.scalars().all()]


def _sync_create_paper(data):
    with create_session() as session:
        paper = PaperAsset(**data)
        session.add(paper)
        session.commit()
        session.refresh(paper)
        return to_dict(paper)


def _sync_get_paper(paper_id):
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(PaperAsset).where(PaperAsset.id == paper_id))
        p = result.scalar_one_or_none()
        return to_dict(p) if p else None


def _sync_update_paper(paper_id, data):
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(PaperAsset).where(PaperAsset.id == paper_id))
        paper = result.scalar_one_or_none()
        if not paper:
            return None
        for key, value in data.items():
            if hasattr(paper, key):
                setattr(paper, key, value)
        session.commit()
        session.refresh(paper)
        return to_dict(paper)


def _sync_list_claims(project_id=None, support_level=None):
    with create_session() as session:
        from sqlalchemy import select
        query = select(ClaimAsset)
        if project_id:
            query = query.where(ClaimAsset.project_id == project_id)
        if support_level:
            query = query.where(ClaimAsset.support_level == support_level)
        query = query.order_by(ClaimAsset.updated_at.desc())
        result = session.execute(query)
        return [to_dict(c) for c in result.scalars().all()]


def _sync_create_claim(data):
    with create_session() as session:
        claim = ClaimAsset(**data)
        session.add(claim)
        session.commit()
        session.refresh(claim)
        return to_dict(claim)


def _sync_update_claim(claim_id, data):
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(ClaimAsset).where(ClaimAsset.id == claim_id))
        claim = result.scalar_one_or_none()
        if not claim:
            return None
        for key, value in data.items():
            if hasattr(claim, key):
                setattr(claim, key, value)
        session.commit()
        session.refresh(claim)
        return to_dict(claim)


def _sync_list_sources():
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(Source).order_by(Source.priority.desc()))
        return [to_dict(s) for s in result.scalars().all()]


def _sync_create_source(data):
    with create_session() as session:
        source = Source(**data)
        session.add(source)
        session.commit()
        session.refresh(source)
        return to_dict(source)


def _sync_update_source(source_id, data):
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(Source).where(Source.id == source_id))
        source = result.scalar_one_or_none()
        if not source:
            return None
        for key, value in data.items():
            if hasattr(source, key):
                setattr(source, key, value)
        session.commit()
        session.refresh(source)
        return to_dict(source)


def _sync_get_today_feed():
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(
            select(FeedItem)
            .where(FeedItem.status.in_(["discovered", "ranked", "presented"]))
            .order_by(FeedItem.rank_score.desc())
            .limit(20)
        )
        return [to_dict(i) for i in result.scalars().all()]


def _sync_list_feed(source_id=None, status=None):
    with create_session() as session:
        from sqlalchemy import select
        query = select(FeedItem)
        if source_id:
            query = query.where(FeedItem.source_id == source_id)
        if status:
            query = query.where(FeedItem.status == status)
        query = query.order_by(FeedItem.rank_score.desc())
        result = session.execute(query)
        return [to_dict(i) for i in result.scalars().all()]


def _sync_create_feed_item(data):
    with create_session() as session:
        item = FeedItem(**data)
        session.add(item)
        session.commit()
        session.refresh(item)
        return to_dict(item)


def _sync_update_feed_item(item_id, data):
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(FeedItem).where(FeedItem.id == item_id))
        item = result.scalar_one_or_none()
        if not item:
            return None
        for key, value in data.items():
            if hasattr(item, key):
                setattr(item, key, value)
        session.commit()
        session.refresh(item)
        return to_dict(item)


# Async wrappers using run_in_executor
loop = asyncio.get_event_loop


async def asyncio_get_projects():
    return await loop().run_in_executor(None, _sync_list_projects)


async def asyncio_create_project(data):
    return await loop().run_in_executor(None, partial(_sync_create_project, data))


async def asyncio_get_project(project_id):
    return await loop().run_in_executor(None, partial(_sync_get_project, project_id))


async def asyncio_list_intents(project_id):
    return await loop().run_in_executor(None, partial(_sync_list_intents, project_id))


async def asyncio_create_intent(data):
    return await loop().run_in_executor(None, partial(_sync_create_intent, data))


async def asyncio_update_intent(intent_id, data):
    return await loop().run_in_executor(None, partial(_sync_update_intent, intent_id, data))


async def asyncio_list_papers(library_status=None, read_status=None):
    return await loop().run_in_executor(None, partial(_sync_list_papers, library_status, read_status))


async def asyncio_create_paper(data):
    return await loop().run_in_executor(None, partial(_sync_create_paper, data))


async def asyncio_get_paper(paper_id):
    return await loop().run_in_executor(None, partial(_sync_get_paper, paper_id))


async def asyncio_update_paper(paper_id, data):
    return await loop().run_in_executor(None, partial(_sync_update_paper, paper_id, data))


async def asyncio_list_claims(project_id=None, support_level=None):
    return await loop().run_in_executor(None, partial(_sync_list_claims, project_id, support_level))


async def asyncio_create_claim(data):
    return await loop().run_in_executor(None, partial(_sync_create_claim, data))


async def asyncio_update_claim(claim_id, data):
    return await loop().run_in_executor(None, partial(_sync_update_claim, claim_id, data))


async def asyncio_list_sources():
    return await loop().run_in_executor(None, _sync_list_sources)


async def asyncio_create_source(data):
    return await loop().run_in_executor(None, partial(_sync_create_source, data))


async def asyncio_update_source(source_id, data):
    return await loop().run_in_executor(None, partial(_sync_update_source, source_id, data))


async def asyncio_get_today_feed():
    return await loop().run_in_executor(None, _sync_get_today_feed)


async def asyncio_list_feed(source_id=None, status=None):
    return await loop().run_in_executor(None, partial(_sync_list_feed, source_id, status))


async def asyncio_create_feed_item(data):
    return await loop().run_in_executor(None, partial(_sync_create_feed_item, data))


async def asyncio_update_feed_item(item_id, data):
    return await loop().run_in_executor(None, partial(_sync_update_feed_item, item_id, data))


# StyleAsset helpers
def _sync_list_styles():
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(StyleAsset).order_by(StyleAsset.updated_at.desc()))
        return [to_dict(s) for s in result.scalars().all()]


def _sync_create_style(data):
    with create_session() as session:
        style = StyleAsset(**data)
        session.add(style)
        session.commit()
        session.refresh(style)
        return to_dict(style)


def _sync_get_style(style_id):
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(StyleAsset).where(StyleAsset.id == style_id))
        s = result.scalar_one_or_none()
        return to_dict(s) if s else None


def _sync_update_style(style_id, data):
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(StyleAsset).where(StyleAsset.id == style_id))
        style = result.scalar_one_or_none()
        if not style:
            return None
        for key, value in data.items():
            if hasattr(style, key):
                setattr(style, key, value)
        session.commit()
        session.refresh(style)
        return to_dict(style)


def _sync_delete_style(style_id):
    with create_session() as session:
        from sqlalchemy import select
        result = session.execute(select(StyleAsset).where(StyleAsset.id == style_id))
        style = result.scalar_one_or_none()
        if not style:
            return None
        session.delete(style)
        session.commit()
        return True


async def asyncio_list_styles():
    return await loop().run_in_executor(None, _sync_list_styles)


async def asyncio_create_style(data):
    return await loop().run_in_executor(None, partial(_sync_create_style, data))


async def asyncio_get_style(style_id):
    return await loop().run_in_executor(None, partial(_sync_get_style, style_id))


async def asyncio_update_style(style_id, data):
    return await loop().run_in_executor(None, partial(_sync_update_style, style_id, data))


async def asyncio_delete_style(style_id):
    return await loop().run_in_executor(None, partial(_sync_delete_style, style_id))
