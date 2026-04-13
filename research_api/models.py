"""Research Workbench SQLAlchemy models."""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from research_api.base import Base


def new_id():
    return str(uuid.uuid4())


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (
        Index("ix_projects_updated_at", "updated_at"),
        Index("ix_projects_status", "status"),
    )

    id = Column(String, primary_key=True, default=new_id)
    title = Column(String, nullable=False)
    goal = Column(String, nullable=True)
    current_direction = Column(String, nullable=True)
    status = Column(String, default="active")  # active, paused, completed
    stage = Column(String, default="文献调研")  # 文献调研, 实验设计, 实验实施, 论文写作, 投稿, 完成
    urgency = Column(String, default="normal")  # urgent, normal, low
    expected_completion = Column(DateTime, nullable=True)  # 预期完成时间
    last_active_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    intents = relationship("Intent", back_populates="project")


class Intent(Base):
    __tablename__ = "intents"

    id = Column(String, primary_key=True, default=new_id)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    intent_type = Column(String, nullable=False)  # LiteratureTracking, Writing, ReviewRebuttal
    title = Column(String, nullable=False)
    goal = Column(String, nullable=True)
    status = Column(String, default="active")  # active, paused, blocked, completed
    priority = Column(Integer, default=1)
    next_action = Column(String, nullable=True)
    risk_flags = Column(JSON, nullable=True)  # List stored as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="intents")


class PaperAsset(Base):
    __tablename__ = "paper_assets"
    __table_args__ = (
        Index("ix_paper_assets_library_status", "library_status"),
        Index("ix_paper_assets_updated_at", "updated_at"),
        Index("ix_paper_assets_project_id", "project_id"),
    )

    id = Column(String, primary_key=True, default=new_id)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True)
    title = Column(String, nullable=False)
    authors_text = Column(String, nullable=True)
    journal_or_source = Column(String, nullable=True)
    published_at = Column(String, nullable=True)
    doi = Column(String, nullable=True)
    abstract = Column(Text, nullable=True)
    source_url = Column(String, nullable=True)
    pdf_url = Column(String, nullable=True)
    local_pdf_path = Column(String, nullable=True)
    quick_read_summary = Column(Text, nullable=True)
    why_relevant = Column(Text, nullable=True)
    potential_use = Column(Text, nullable=True)
    read_status = Column(String, default="unread")  # unread, quick-reviewed, skimmed, deeply-read
    library_status = Column(String, default="pending")  # pending, library
    style_candidate = Column(Boolean, default=False)
    asset_type = Column(String, default="paper")  # paper, figure, table, code, dataset
    tags = Column(JSON, nullable=True)  # List of tags stored as JSON
    local_path = Column(String, nullable=True)  # Path to local MinerU folder or asset file
    content_text = Column(Text, nullable=True)  # Extracted text/markdown content for papers
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ClaimAsset(Base):
    __tablename__ = "claim_assets"
    __table_args__ = (
        Index("ix_claim_assets_project_id", "project_id"),
        Index("ix_claim_assets_support_level", "support_level"),
    )

    id = Column(String, primary_key=True, default=new_id)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    claim_text = Column(Text, nullable=False)
    claim_type = Column(String, nullable=True)  # performance, robustness, generalization
    support_level = Column(String, default="unsupported")  # unsupported, weakly_supported, partially_supported, supported, contested
    supporting_experiment_refs = Column(JSON, nullable=True)
    supporting_figure_refs = Column(JSON, nullable=True)
    supporting_paper_refs = Column(JSON, nullable=True)
    linked_sections = Column(JSON, nullable=True)
    open_caveats = Column(Text, nullable=True)
    status = Column(String, default="draft")  # draft, partially-supported, supported, disputed, removed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Source(Base):
    __tablename__ = "sources"

    id = Column(String, primary_key=True, default=new_id)
    name = Column(String, nullable=False)
    category = Column(String, nullable=True)  # journal, arxiv, conference
    intake_type = Column(String, default="rss")  # rss, toc, api
    feed_url = Column(String, nullable=True)
    site_url = Column(String, nullable=True)
    priority = Column(Integer, default=1)
    enabled = Column(Boolean, default=True)
    include_in_brief = Column(Boolean, default=True)
    allow_pdf_attempt = Column(Boolean, default=False)
    topic_bias = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FeedItem(Base):
    __tablename__ = "feed_items"
    __table_args__ = (
        Index("ix_feed_items_status", "status"),
        Index("ix_feed_items_source_id", "source_id"),
    )

    id = Column(String, primary_key=True, default=new_id)
    source_id = Column(String, ForeignKey("sources.id"), nullable=True)
    external_key = Column(String, nullable=True)
    title = Column(String, nullable=False)
    title_zh = Column(String, nullable=True)  # 中文标题
    authors_text = Column(String, nullable=True)
    published_at = Column(String, nullable=True)
    abstract = Column(Text, nullable=True)
    abstract_zh = Column(Text, nullable=True)  # 中文摘要
    source_url = Column(String, nullable=True)
    pdf_url = Column(String, nullable=True)
    doi = Column(String, nullable=True)
    rank_score = Column(Float, default=0.0)
    novelty_score = Column(Float, default=0.0)
    transferability_score = Column(Float, default=0.0)
    status = Column(String, default="discovered")  # discovered, ranked, presented, quick-reviewed, saved, ignored
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StyleAsset(Base):
    __tablename__ = "style_assets"

    id = Column(String, primary_key=True, default=new_id)
    title = Column(String, nullable=False)
    abstract_pattern = Column(String)
    intro_pattern = Column(String)
    methods_pattern = Column(String)
    tone_notes = Column(String)
    citation_format = Column(String)
    source_paper_refs = Column(String)  # JSON string
    usage_notes = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
