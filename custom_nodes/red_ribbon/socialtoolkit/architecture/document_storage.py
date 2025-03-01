from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DocumentStatus(str, Enum):
    NEW = "new"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"

class VersionStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"

class SourceType(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"

class DocumentStorageConfigs(BaseModel):
    """Configuration for Document Storage"""
    database_connection_string: str
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    batch_size: int = 100
    vector_dim: int = 1536  # Common dimension for embeddings like OpenAI's
    storage_type: str = "sql"  # Alternatives: "nosql", "in_memory", etc.

class DocumentStorage:
    """
    Document Storage system based on mermaid ER diagram in README.md
    Manages the storage and retrieval of documents, versions, metadata, and vectors
    """
    
    def __init__(self, resources: Dict[str, Any], configs: DocumentStorageConfigs):
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including storage services
            configs: Configuration for Document Storage
        """
        self.resources = resources
        self.configs = configs
        
        # Extract needed services from resources
        self.db_service = resources.get("database_service")
        self.cache_service = resources.get("cache_service")
        self.vector_store = resources.get("vector_store_service")
        self.id_generator = resources.get("id_generator_service", self._generate_uuid)
        
        logger.info("DocumentStorage initialized with services")

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute document storage operations based on the action
        
        Args:
            action: Operation to perform (store, retrieve, update, delete)
            **kwargs: Operation-specific parameters
            
        Returns:
            Dictionary containing operation results
        """
        logger.info(f"Starting document storage operation: {action}")
        
        if action == "store":
            return self._store_documents(
                kwargs.get("documents", []), 
                kwargs.get("metadata", []), 
                kwargs.get("vectors", [])
            )
        elif action == "retrieve":
            return self._retrieve_documents(
                document_ids=kwargs.get("document_ids", []),
                filters=kwargs.get("filters", {})
            )
        elif action == "update":
            return self._update_documents(
                kwargs.get("documents", [])
            )
        elif action == "delete":
            return self._delete_documents(
                kwargs.get("document_ids", [])
            )
        elif action == "get_vectors":
            return self._get_vectors(
                kwargs.get("document_ids", [])
            )
        else:
            logger.error(f"Unknown action: {action}")
            raise ValueError(f"Unknown action: {action}")

    def store(self, documents: List[Any], metadata: List[Any], vectors: List[Any]) -> Dict[str, Any]:
        """
        Store documents, metadata, and vectors
        
        Args:
            documents: Documents to store
            metadata: Metadata for the documents
            vectors: Vectors for the documents
            
        Returns:
            Dictionary with storage status
        """
        return self.control_flow("store", documents=documents, metadata=metadata, vectors=vectors)

    def get_documents_and_vectors(self, document_ids: List[str] = None, 
                                 filters: Dict[str, Any] = None) -> Tuple[List[Any], List[Any]]:
        """
        Retrieve documents and their vectors
        
        Args:
            document_ids: Optional list of document IDs to retrieve
            filters: Optional filters to apply
            
        Returns:
            Tuple of (documents, vectors)
        """
        result = self.control_flow(
            "retrieve", document_ids=document_ids, filters=filters
        )
        documents = result.get("documents", [])
        
        vectors_result = self.control_flow(
            "get_vectors", document_ids=[doc.get("id") for doc in documents]
        )
        vectors = vectors_result.get("vectors", [])
        
        return documents, vectors
    
    def _store_documents(self, documents: List[Any], metadata: List[Any], vectors: List[Any]) -> Dict[str, Any]:
        """Store documents, metadata, and vectors in the database"""
        try:
            # 1. Store source information if needed
            source_ids = self._store_sources(documents)
            
            # 2. Store documents
            document_ids = self._store_document_entries(documents, source_ids)
            
            # 3. Create versions for documents
            version_ids = self._create_versions(document_ids)
            
            # 4. Store metadata
            self._store_metadata(metadata, document_ids)
            
            # 5. Store content
            content_ids = self._store_content(documents, version_ids)
            
            # 6. Create version-content associations
            self._create_version_content_links(version_ids, content_ids)
            
            # 7. Store vectors
            self._store_vectors(vectors, content_ids)
            
            return {
                "success": True,
                "document_ids": document_ids,
                "version_ids": version_ids,
                "content_ids": content_ids
            }
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _retrieve_documents(self, document_ids: List[str] = None, 
                           filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Retrieve documents from the database"""
        try:
            documents = []
            
            # Use document IDs if provided, otherwise use filters
            if document_ids:
                query = f"SELECT * FROM Documents WHERE document_id IN ({','.join(['?']*len(document_ids))})"
                documents = self.db_service.execute(query, document_ids)
            elif filters:
                # Build WHERE clause based on filters
                where_clauses = []
                params = []
                
                for key, value in filters.items():
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                
                query = f"SELECT * FROM Documents WHERE {' AND '.join(where_clauses)}"
                documents = self.db_service.execute(query, params)
            else:
                # Retrieve all documents (with limit)
                query = f"SELECT * FROM Documents LIMIT {self.configs.batch_size}"
                documents = self.db_service.execute(query)
            
            return {
                "success": True,
                "documents": documents
            }
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _update_documents(self, documents: List[Any]) -> Dict[str, Any]:
        """Update existing documents"""
        # Implementation for updating documents
        return {"success": True}
        
    def _delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """Delete documents by ID"""
        # Implementation for deleting documents
        return {"success": True}
        
    def _get_vectors(self, document_ids: List[str]) -> Dict[str, Any]:
        """Get vectors for the specified document IDs"""
        try:
            # Get content IDs for the documents
            content_ids_query = """
                SELECT c.content_id FROM Contents c
                JOIN VersionsContents vc ON c.content_id = vc.content_id
                JOIN Versions v ON vc.version_id = v.version_id
                JOIN Documents d ON v.document_id = d.document_id
                WHERE d.document_id IN ({}) AND v.current_version = 1
            """.format(','.join(['?']*len(document_ids)))
            
            content_ids_result = self.db_service.execute(content_ids_query, document_ids)
            content_ids = [r["content_id"] for r in content_ids_result]
            
            # Get vectors for the content
            vectors_query = f"""
                SELECT * FROM Vectors WHERE content_id IN ({','.join(['?']*len(content_ids))})
            """
            vectors = self.db_service.execute(vectors_query, content_ids)
            
            return {
                "success": True,
                "vectors": vectors
            }
        except Exception as e:
            logger.error(f"Error retrieving vectors: {e}")
            return {
                "success": False,
                "error": str(e),
                "vectors": []
            }
    
    # Helper methods for database operations
    def _store_sources(self, documents: List[Any]) -> Dict[str, str]:
        """Store sources and return a mapping of URL to source_id"""
        source_map = {}
        for doc in documents:
            url = doc.get("url", "")
            domain = self._extract_domain(url)
            
            if domain not in source_map:
                source_id = self.id_generator()
                
                # Check if source already exists
                query = "SELECT id FROM Sources WHERE id = ?"
                result = self.db_service.execute(query, [domain])
                
                if not result:
                    # Insert new source
                    insert_query = "INSERT INTO Sources (id) VALUES (?)"
                    self.db_service.execute(insert_query, [domain])
                
                source_map[domain] = domain  # Source ID is the domain
        
        return source_map
    
    def _store_document_entries(self, documents: List[Any], source_ids: Dict[str, str]) -> List[str]:
        """Store document entries and return document IDs"""
        document_ids = []
        
        for doc in documents:
            url = doc.get("url", "")
            domain = self._extract_domain(url)
            source_id = source_ids.get(domain)
            
            document_id = self.id_generator()
            document_type = self._determine_document_type(url)
            
            # Insert document
            insert_query = """
                INSERT INTO Documents (
                    document_id, source_id, url, document_type,
                    status, priority
                ) VALUES (?, ?, ?, ?, ?, ?)
            """
            
            params = [
                document_id, 
                source_id, 
                url, 
                document_type,
                DocumentStatus.NEW.value,
                5  # Default priority
            ]
            
            self.db_service.execute(insert_query, params)
            document_ids.append(document_id)
        
        return document_ids
    
    def _create_versions(self, document_ids: List[str]) -> List[str]:
        """Create initial versions for documents"""
        version_ids = []
        
        for document_id in document_ids:
            version_id = self.id_generator()
            
            # Insert version
            insert_query = """
                INSERT INTO Versions (
                    version_id, document_id, current_version,
                    version_number, status, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """
            
            params = [
                version_id,
                document_id,
                True,  # Current version
                "1.0",  # Initial version
                VersionStatus.ACTIVE.value,
                datetime.now()
            ]
            
            self.db_service.execute(insert_query, params)
            
            # Update document with current version ID
            update_query = """
                UPDATE Documents
                SET current_version_id = ?, status = ?
                WHERE document_id = ?
            """
            
            update_params = [
                version_id,
                DocumentStatus.COMPLETE.value,
                document_id
            ]
            
            self.db_service.execute(update_query, update_params)
            version_ids.append(version_id)
        
        return version_ids
    
    def _store_metadata(self, metadata_list: List[Any], document_ids: List[str]) -> None:
        """Store metadata for documents"""
        for i, metadata in enumerate(metadata_list):
            if i >= len(document_ids):
                break
                
            document_id = document_ids[i]
            metadata_id = self.id_generator()
            
            # Insert metadata
            insert_query = """
                INSERT INTO Metadatas (
                    metadata_id, document_id, other_metadata,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
            """
            
            params = [
                metadata_id,
                document_id,
                metadata.get("metadata", "{}"),
                datetime.now(),
                datetime.now()
            ]
            
            self.db_service.execute(insert_query, params)
    
    def _store_content(self, documents: List[Any], version_ids: List[str]) -> List[str]:
        """Store content for document versions"""
        content_ids = []
        
        for i, doc in enumerate(documents):
            if i >= len(version_ids):
                break
                
            version_id = version_ids[i]
            content_id = self.id_generator()
            
            # Insert content
            insert_query = """
                INSERT INTO Contents (
                    content_id, version_id, raw_content,
                    processed_content, hash
                ) VALUES (?, ?, ?, ?, ?)
            """
            
            content = doc.get("content", "")
            processed_content = content  # In reality, this might go through processing
            content_hash = self._generate_hash(content)
            
            params = [
                content_id,
                version_id,
                content,
                processed_content,
                content_hash
            ]
            
            self.db_service.execute(insert_query, params)
            content_ids.append(content_id)
        
        return content_ids
    
    def _create_version_content_links(self, version_ids: List[str], content_ids: List[str]) -> None:
        """Create links between versions and content"""
        for i, version_id in enumerate(version_ids):
            if i >= len(content_ids):
                break
                
            content_id = content_ids[i]
            
            # Insert version-content link
            insert_query = """
                INSERT INTO VersionsContents (
                    version_id, content_id, created_at, source_type
                ) VALUES (?, ?, ?, ?)
            """
            
            params = [
                version_id,
                content_id,
                datetime.now(),
                SourceType.PRIMARY.value
            ]
            
            self.db_service.execute(insert_query, params)
    
    def _store_vectors(self, vectors: List[Any], content_ids: List[str]) -> None:
        """Store vectors for content"""
        for i, vector in enumerate(vectors):
            if i >= len(content_ids):
                break
                
            content_id = content_ids[i]
            vector_id = self.id_generator()
            
            # Insert vector
            insert_query = """
                INSERT INTO Vectors (
                    vector_id, content_id, vector_embedding, embedding_type
                ) VALUES (?, ?, ?, ?)
            """
            
            params = [
                vector_id,
                content_id,
                vector.get("embedding"),
                vector.get("embedding_type", "default")
            ]
            
            self.db_service.execute(insert_query, params)
    
    # Utility methods
    def _generate_uuid(self) -> str:
        """Generate a UUID string"""
        return str(uuid4())
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        import re
        match = re.search(r'https?://([^/]+)', url)
        return match.group(1) if match else url
    
    def _determine_document_type(self, url: str) -> str:
        """Determine document type from URL"""
        if url.endswith('.pdf'):
            return 'pdf'
        elif url.endswith('.doc') or url.endswith('.docx'):
            return 'word'
        else:
            return 'html'
    
    def _generate_hash(self, content: str) -> str:
        """Generate a hash for content"""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()