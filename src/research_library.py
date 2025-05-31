"""
Research Library Manager
Manages the collection of research documents, papers, and findings
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResearchLibrary:
    """
    Manages a library of research documents like a real lab.
    Organizes papers, findings, protocols, and data.
    """
    
    def __init__(self, library_dir: Path):
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.sections = {
            'papers': self.library_dir / 'papers',
            'protocols': self.library_dir / 'protocols',
            'data': self.library_dir / 'data',
            'findings': self.library_dir / 'findings',
            'hypotheses': self.library_dir / 'hypotheses',
            'innovations': self.library_dir / 'innovations',
            'reports': self.library_dir / 'reports'
        }
        
        for section in self.sections.values():
            section.mkdir(exist_ok=True)
            
        # Create library index
        self.index_path = self.library_dir / 'library_index.json'
        self._load_index()
        
    def _load_index(self):
        """Load or create the library index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                'documents': {},
                'projects': {},
                'tags': {},
                'citations': {},
                'last_updated': datetime.now().isoformat()
            }
            self._save_index()
            
    def _save_index(self):
        """Save the library index."""
        self.index['last_updated'] = datetime.now().isoformat()
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
            
    async def save_document(self, project_name: str, doc_type: str, content: Dict[str, Any], filename: str = None) -> Path:
        """
        Save a research document to the library.
        
        Args:
            project_name: Name of the research project
            doc_type: Type of document (paper, protocol, data, etc.)
            content: Document content as dictionary
            filename: Optional custom filename
            
        Returns:
            Path to saved document
        """
        # Determine section
        section = self._get_section_for_type(doc_type)
        
        # Create project subdirectory
        project_dir = section / project_name
        project_dir.mkdir(exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{doc_type}_{timestamp}.json"
            
        # Save document
        doc_path = project_dir / filename
        with open(doc_path, 'w') as f:
            json.dump(content, f, indent=2)
            
        # Update index
        doc_id = f"{project_name}/{doc_type}/{filename}"
        self.index['documents'][doc_id] = {
            'path': str(doc_path.relative_to(self.library_dir)),
            'project': project_name,
            'type': doc_type,
            'created': datetime.now().isoformat(),
            'size': doc_path.stat().st_size,
            'metadata': content.get('metadata', {})
        }
        
        # Update project index
        if project_name not in self.index['projects']:
            self.index['projects'][project_name] = {
                'created': datetime.now().isoformat(),
                'documents': []
            }
        self.index['projects'][project_name]['documents'].append(doc_id)
        
        # Extract and index tags
        tags = content.get('tags', [])
        if isinstance(content, dict) and 'hypothesis' in content:
            tags.append('hypothesis')
        if 'mechanism' in str(content).lower():
            tags.append('mechanism')
            
        for tag in tags:
            if tag not in self.index['tags']:
                self.index['tags'][tag] = []
            self.index['tags'][tag].append(doc_id)
            
        self._save_index()
        logger.info(f"Saved document: {doc_id}")
        
        return doc_path
        
    def _get_section_for_type(self, doc_type: str) -> Path:
        """Get the appropriate section directory for a document type."""
        type_mapping = {
            'problem_definition': 'hypotheses',
            'hypothesis': 'hypotheses',
            'literature_review': 'papers',
            'experiment': 'data',
            'trial_protocol': 'protocols',
            'innovation_proposal': 'innovations',
            'final_report': 'reports',
            'findings': 'findings'
        }
        
        section_name = type_mapping.get(doc_type, 'data')
        return self.sections.get(section_name, self.sections['data'])
        
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        if doc_id not in self.index['documents']:
            return None
            
        doc_info = self.index['documents'][doc_id]
        doc_path = self.library_dir / doc_info['path']
        
        if not doc_path.exists():
            logger.error(f"Document file missing: {doc_path}")
            return None
            
        with open(doc_path, 'r') as f:
            content = json.load(f)
            
        return {
            'id': doc_id,
            'content': content,
            'metadata': doc_info
        }
        
    async def search_documents(self, query: str = None, project: str = None, doc_type: str = None, tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search documents in the library.
        
        Args:
            query: Text to search for in documents
            project: Filter by project name
            doc_type: Filter by document type
            tags: Filter by tags
            
        Returns:
            List of matching documents
        """
        results = []
        
        for doc_id, doc_info in self.index['documents'].items():
            # Apply filters
            if project and doc_info['project'] != project:
                continue
            if doc_type and doc_info['type'] != doc_type:
                continue
                
            # Check tags
            if tags:
                doc_tags = []
                for tag, tagged_docs in self.index['tags'].items():
                    if doc_id in tagged_docs:
                        doc_tags.append(tag)
                        
                if not any(tag in doc_tags for tag in tags):
                    continue
                    
            # Load document for query search
            if query:
                doc = await self.get_document(doc_id)
                if doc and query.lower() not in json.dumps(doc['content']).lower():
                    continue
                    
            results.append({
                'id': doc_id,
                'info': doc_info
            })
            
        return results
        
    async def create_bibliography(self, project_name: str) -> str:
        """Create a bibliography of all sources for a project."""
        docs = await self.search_documents(project=project_name, doc_type='literature_review')
        
        bibliography = f"# Bibliography: {project_name}\n\n"
        bibliography += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        citations = []
        for doc_data in docs:
            doc = await self.get_document(doc_data['id'])
            if doc and 'content' in doc:
                content = doc['content']
                if 'synthesis' in content:
                    # Extract any citations from synthesis
                    citations.append({
                        'title': content.get('hypothesis', 'Unknown'),
                        'type': 'Literature Review',
                        'date': doc_data['info']['created']
                    })
                    
        # Format citations
        for i, citation in enumerate(citations, 1):
            bibliography += f"{i}. {citation['title']} ({citation['type']}, {citation['date']})\n"
            
        # Save bibliography
        bib_path = self.sections['reports'] / project_name / 'bibliography.md'
        bib_path.parent.mkdir(exist_ok=True)
        with open(bib_path, 'w') as f:
            f.write(bibliography)
            
        return bibliography
        
    async def get_project_summary(self, project_name: str) -> Dict[str, Any]:
        """Get a summary of all documents for a project."""
        if project_name not in self.index['projects']:
            return {'error': f'Project {project_name} not found'}
            
        project_info = self.index['projects'][project_name]
        doc_ids = project_info['documents']
        
        # Count document types
        doc_types = {}
        total_size = 0
        
        for doc_id in doc_ids:
            if doc_id in self.index['documents']:
                doc_info = self.index['documents'][doc_id]
                doc_type = doc_info['type']
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                total_size += doc_info.get('size', 0)
                
        return {
            'project': project_name,
            'created': project_info['created'],
            'total_documents': len(doc_ids),
            'document_types': doc_types,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2)
        }
        
    async def export_project(self, project_name: str, export_path: Path = None) -> Path:
        """Export all documents for a project to a zip file."""
        if not export_path:
            export_path = self.library_dir / 'exports'
            export_path.mkdir(exist_ok=True)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_file = export_path / f"{project_name}_export_{timestamp}"
        
        # Create temporary directory for export
        temp_dir = export_path / f"temp_{timestamp}"
        temp_dir.mkdir()
        
        try:
            # Copy all project documents
            for section in self.sections.values():
                project_dir = section / project_name
                if project_dir.exists():
                    dest_section = temp_dir / section.name
                    dest_section.mkdir()
                    shutil.copytree(project_dir, dest_section / project_name)
                    
            # Create project summary
            summary = await self.get_project_summary(project_name)
            summary_path = temp_dir / 'project_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            # Create bibliography
            bibliography = await self.create_bibliography(project_name)
            bib_path = temp_dir / 'bibliography.md'
            with open(bib_path, 'w') as f:
                f.write(bibliography)
                
            # Create zip archive
            shutil.make_archive(str(export_file), 'zip', temp_dir)
            
            return Path(f"{export_file}.zip")
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
    def get_library_stats(self) -> Dict[str, Any]:
        """Get overall library statistics."""
        total_docs = len(self.index['documents'])
        total_projects = len(self.index['projects'])
        total_tags = len(self.index['tags'])
        
        # Count documents by type
        doc_by_type = {}
        total_size = 0
        
        for doc_info in self.index['documents'].values():
            doc_type = doc_info['type']
            doc_by_type[doc_type] = doc_by_type.get(doc_type, 0) + 1
            total_size += doc_info.get('size', 0)
            
        return {
            'total_documents': total_docs,
            'total_projects': total_projects,
            'total_tags': total_tags,
            'documents_by_type': doc_by_type,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'last_updated': self.index['last_updated']
        }
        
    async def add_citation(self, doc_id: str, citation: Dict[str, str]):
        """Add a citation reference to a document."""
        if doc_id not in self.index['documents']:
            return False
            
        citation_id = f"cite_{len(self.index['citations'])}"
        self.index['citations'][citation_id] = {
            'document': doc_id,
            'citation': citation,
            'added': datetime.now().isoformat()
        }
        
        self._save_index()
        return True
        
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects in the library."""
        projects = []
        for name, info in self.index['projects'].items():
            projects.append({
                'name': name,
                'created': info['created'],
                'document_count': len(info['documents'])
            })
        return sorted(projects, key=lambda x: x['created'], reverse=True) 