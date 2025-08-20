# ci_vector_processor.py
# Enhanced version of your repo processor for CI/CD environments
# Handles incremental updates, multiple repos, and Slack files

import os
import sys
import git
import shutil
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from datetime import datetime
import logging

# Pinecone, Anthropic, and OpenAI imports
try:
    import pinecone
    from anthropic import Anthropic
    from openai import OpenAI  # Keep for embeddings
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("Required packages not available. Install with: pip install pinecone-client anthropic openai")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CICDVectorProcessor:
    def __init__(self, 
                 chroma_path: str = "./chroma_db",
                 collection_name: str = "github_repo",
                 use_pinecone: bool = False):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.use_pinecone = use_pinecone
        
        # Initialize ChromaDB (always for local processing)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Initialize Pinecone if requested
        self.pinecone_index = None
        self.anthropic_client = None
        self.openai_client = None  # Keep for embeddings
        
        if use_pinecone and PINECONE_AVAILABLE:
            self._setup_pinecone()
        
        # Track processing metadata
        self.processing_log = []
    
    def _setup_pinecone(self):
        """Initialize Pinecone connection."""
        try:
            api_key = os.getenv('PINECONE_API_KEY')
            environment = os.getenv('PINECONE_ENVIRONMENT')
            index_name = os.getenv('PINECONE_INDEX_NAME')
            
            if not all([api_key, index_name]):
                logger.error("Missing Pinecone credentials in environment variables")
                return
            
            # Initialize Pinecone with new SDK
            pc = Pinecone(api_key=api_key)
            
            # Connect to index
            if index_name not in [index.name for index in pc.list_indexes()]:
                logger.error(f"Pinecone index '{index_name}' not found")
                return
            
            self.pinecone_index = pc.Index(index_name)
            
            # Initialize OpenAI for embeddings
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)
            else:
                logger.warning("No OpenAI API key found - will use ChromaDB embeddings")
            
            logger.info(f"Successfully connected to Pinecone index: {index_name}")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone: {e}")
            self.pinecone_index = None
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI (Anthropic doesn't have embeddings API)."""
        if self.openai_client:
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Error getting OpenAI embedding: {e}")
                return None
        else:
            # Use ChromaDB's default embedding function as fallback
            logger.warning("No embedding service available - using ChromaDB fallback")
            return None
    
    def process_all_files(self, data_directory: str = "./data") -> Dict[str, Any]:
        """Process all files in the data directory - repos, Slack files, everything."""
        data_path = Path(data_directory)
        
        if not data_path.exists():
            logger.warning(f"Data directory doesn't exist: {data_directory}")
            return {'success': False, 'error': 'Data directory not found'}
        
        documents = []
        metadatas = []
        ids = []
        vectors_for_pinecone = []
        processed_files = 0
        
        # Process all files recursively
        for file_path in data_path.rglob("*"):
            if not (file_path.is_file() and self.should_process_file(file_path)):
                continue
            
            try:
                content = self.get_file_content(file_path)
                if not content.strip():
                    continue
                
                # Determine source type
                relative_path = file_path.relative_to(data_path)
                source_type = self._determine_source_type(relative_path)
                
                chunks = self.get_chunks(content)
                file_hash = self.get_file_hash(file_path)
                
                for i, chunk in enumerate(chunks):
                    chunk_id = hashlib.md5(
                        f"{file_path}_{i}_{file_hash}".encode()
                    ).hexdigest()
                    
                    metadata = {
                        'file_path': str(relative_path),
                        'chunk_index': i,
                        'filetype': file_path.suffix,
                        'source_type': source_type,
                        'file_hash': file_hash,
                        'processed_at': datetime.now().isoformat(),
                        'text': chunk  # Include text in metadata for Pinecone
                    }
                    
                    documents.append(chunk)
                    metadatas.append(metadata)
                    ids.append(chunk_id)
                    
                    # Prepare for Pinecone if enabled
                    if self.use_pinecone and self.pinecone_index:
                        embedding = self._get_embedding(chunk)
                        if embedding:
                            vectors_for_pinecone.append({
                                'id': chunk_id,
                                'values': embedding,
                                'metadata': metadata
                            })
                
                processed_files += 1
                logger.info(f"Processed: {relative_path}")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        # Add to ChromaDB (local storage)
        if documents:
            logger.info(f"Adding {len(documents)} chunks to ChromaDB")
            # Fixed version - add this at line 177 instead:
            batch_size = 5000
            total_docs = len(documents)
            print(f"Processing {total_docs} documents in batches of {batch_size}")

            for i in range(0, total_docs, batch_size):
                end_idx = min(i + batch_size, total_docs)
                
                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx] if metadatas else None,
                    ids=ids[i:end_idx]
                )
                print(f"Added batch {i//batch_size + 1}: documents {i+1}-{end_idx}")

        
        # Upload to Pinecone if enabled
        pinecone_success = False
        if self.use_pinecone and vectors_for_pinecone and self.pinecone_index:
            pinecone_success = self._upload_to_pinecone(vectors_for_pinecone)
        
        return {
            'success': True,
            'processed_files': processed_files,
            'total_chunks': len(documents),
            'pinecone_uploaded': pinecone_success,
            'pinecone_chunks': len(vectors_for_pinecone) if vectors_for_pinecone else 0
        }
    
    def _upload_to_pinecone(self, vectors: List[Dict]) -> bool:
        """Upload vectors to Pinecone in batches."""
        try:
            batch_size = 100  # Pinecone batch limit
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                
                # Convert to Pinecone format
                pinecone_vectors = [
                    (v['id'], v['values'], v['metadata'])
                    for v in batch
                ]
                
                self.pinecone_index.upsert(vectors=pinecone_vectors)
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1} to Pinecone")
            
            logger.info(f"Successfully uploaded {len(vectors)} vectors to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to Pinecone: {e}")
            return False
    
    def _determine_source_type(self, relative_path: Path) -> str:
        """Determine the source type based on file path."""
        parts = relative_path.parts
        if 'slack_files' in parts:
            return 'slack'
        elif 'github_repos' in parts:
            return 'github'
        elif any(part.endswith('.git') for part in parts):
            return 'github'
        else:
            return 'manual_upload'
        
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file content for change detection."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return ""
    
    def should_process_file(self, file_path: Path) -> bool:
        """Enhanced file filtering with better CI/CD considerations."""
        skip_extensions = {
            '.idx', '.pack', '.pyc', '.rev', '.sample',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
            '.pdf', '.zip', '.tar', '.gz', '.rar', '.7z',
            '.mp4', '.avi', '.mov', '.wmv', '.mp3', '.wav',
            '.exe', '.dll', '.so', '.dylib', '.bin'
        }
        
        skip_folders = {
            'node_modules', '.git', '__pycache__', '.venv', 'venv', 
            'env', 'wiki_data', '.github', 'target', 'build', 'dist'
        }
        
        # Check folder exclusions
        for part in file_path.parts:
            if part in skip_folders:
                return False
        
        # Check extension exclusions
        if file_path.suffix.lower() in skip_extensions:
            return False
        
        # Check file size (skip very large files > 10MB)
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:
                logger.warning(f"Skipping large file: {file_path} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
                return False
        except:
            pass
        
        return True
    
    def get_file_content(self, file_path: Path) -> str:
        """Read file content with better encoding handling."""
        encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return ""
        
        logger.warning(f"Could not decode file: {file_path}")
        return ""
    
    def get_chunks(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Enhanced chunking with better overlap handling."""
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + chunk_size, len(content))
            
            # Try to break at sentence boundaries
            if end < len(content):
                sentence_break = content.rfind('.', start, end)
                if sentence_break > start + chunk_size // 2:
                    end = sentence_break + 1
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(content) else end
        
        return chunks
    
    def process_repo_incremental(self, 
                                repo_url: str, 
                                changed_files: Optional[List[str]] = None,
                                force_rebuild: bool = False) -> Dict[str, Any]:
        """Process repository with incremental updates."""
        
        clone_path = f"./temp_repo_{hashlib.md5(repo_url.encode()).hexdigest()[:8]}"
        
        try:
            # Clone or update repository
            if os.path.exists(clone_path):
                logger.info(f"Repository exists, pulling latest changes: {clone_path}")
                repo = git.Repo(clone_path)
                repo.remotes.origin.pull()
            else:
                logger.info(f"Cloning repository: {repo_url}")
                git.Repo.clone_from(repo_url, clone_path)
            
            repo_path = Path(clone_path)
            
            # Determine files to process
            if force_rebuild or not changed_files:
                files_to_process = list(repo_path.rglob("*"))
                logger.info("Processing all files (full rebuild)")
            else:
                files_to_process = []
                for changed_file in changed_files:
                    file_path = repo_path / changed_file
                    if file_path.exists():
                        files_to_process.append(file_path)
                logger.info(f"Processing {len(files_to_process)} changed files")
            
            # Process files
            documents = []
            metadatas = []
            ids = []
            processed_files = 0
            
            for file_path in files_to_process:
                if not (file_path.is_file() and self.should_process_file(file_path)):
                    continue
                
                try:
                    relative_path = file_path.relative_to(repo_path)
                    content = self.get_file_content(file_path)
                    
                    if not content.strip():
                        continue
                    
                    # Remove old chunks for this file if doing incremental update
                    if not force_rebuild:
                        self._remove_file_chunks(str(relative_path), repo_url)
                    
                    chunks = self.get_chunks(content)
                    file_hash = self.get_file_hash(file_path)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_id = hashlib.md5(
                            f"{repo_url}_{relative_path}_{i}_{file_hash}".encode()
                        ).hexdigest()
                        
                        documents.append(chunk)
                        metadatas.append({
                            'file_path': str(relative_path),
                            'chunk_index': i,
                            'filetype': file_path.suffix,
                            'repo_url': repo_url,
                            'file_hash': file_hash,
                            'processed_at': datetime.now().isoformat()
                        })
                        ids.append(chunk_id)
                    
                    processed_files += 1
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
            
            # Add to ChromaDB
            if documents:
                logger.info(f"Adding {len(documents)} chunks to ChromaDB")
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            return {
                'success': True,
                'processed_files': processed_files,
                'total_chunks': len(documents),
                'repo_url': repo_url
            }
            
        except Exception as e:
            logger.error(f"Error processing repository {repo_url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'repo_url': repo_url
            }
            
        finally:
            # Cleanup
            if os.path.exists(clone_path):
                shutil.rmtree(clone_path)
    
    def _remove_file_chunks(self, file_path: str, repo_url: str):
        """Remove existing chunks for a file to avoid duplicates."""
        try:
            # ChromaDB doesn't have a direct way to delete by metadata
            # This is a limitation we'd need to work around
            # For now, we'll use unique IDs to avoid duplicates
            pass
        except Exception as e:
            logger.error(f"Error removing old chunks: {e}")
    
    def process_slack_files(self, slack_files_dir: str = "./slack_files") -> Dict[str, Any]:
        """Process files downloaded from Slack."""
        if not os.path.exists(slack_files_dir):
            return {'success': True, 'processed_files': 0, 'message': 'No slack files directory'}
        
        slack_path = Path(slack_files_dir)
        documents = []
        metadatas = []
        ids = []
        processed_files = 0
        
        for file_path in slack_path.rglob("*"):
            if not (file_path.is_file() and self.should_process_file(file_path)):
                continue
            
            try:
                content = self.get_file_content(file_path)
                if not content.strip():
                    continue
                
                chunks = self.get_chunks(content)
                file_hash = self.get_file_hash(file_path)
                
                for i, chunk in enumerate(chunks):
                    chunk_id = hashlib.md5(
                        f"slack_{file_path.name}_{i}_{file_hash}".encode()
                    ).hexdigest()
                    
                    documents.append(chunk)
                    metadatas.append({
                        'file_path': str(file_path.relative_to(slack_path)),
                        'chunk_index': i,
                        'filetype': file_path.suffix,
                        'source': 'slack',
                        'file_hash': file_hash,
                        'processed_at': datetime.now().isoformat()
                    })
                    ids.append(chunk_id)
                
                processed_files += 1
                
            except Exception as e:
                logger.error(f"Error processing Slack file {file_path}: {e}")
                continue
        
        if documents:
            logger.info(f"Adding {len(documents)} Slack chunks to ChromaDB")
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return {
            'success': True,
            'processed_files': processed_files,
            'total_chunks': len(documents)
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        try:
            count = self.collection.count()
            # Get sample of metadata to understand content
            sample = self.collection.get(limit=10)
            
            sources = set()
            file_types = set()
            
            for metadata in sample['metadatas']:
                if 'repo_url' in metadata:
                    sources.add(metadata['repo_url'])
                elif 'source' in metadata:
                    sources.add(metadata['source'])
                
                if 'filetype' in metadata:
                    file_types.add(metadata['filetype'])
            
            return {
                'total_chunks': count,
                'sample_sources': list(sources),
                'sample_file_types': list(file_types)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='CI/CD Vector Processor')
    parser.add_argument('--changed-files', type=str, help='JSON string of changed files')
    parser.add_argument('--force-rebuild', type=str, default='false', help='Force full rebuild')
    parser.add_argument('--data-dir', type=str, default='./data', help='Directory containing all files to process')
    parser.add_argument('--upload-to-pinecone', action='store_true', help='Upload vectors to Pinecone')
    
    args = parser.parse_args()
    
    processor = CICDVectorProcessor(use_pinecone=args.upload_to_pinecone)
    
    force_rebuild = args.force_rebuild.lower() == 'true'
    
    if force_rebuild:
        logger.info("Force rebuild requested - processing all files")
        result = processor.process_all_files(args.data_dir)
    else:
        # Process only changed files if specified
        changed_files = json.loads(args.changed_files) if args.changed_files else None
        if changed_files:
            logger.info(f"Processing {len(changed_files)} changed files")
            # For now, process all files (you can implement incremental later)
            result = processor.process_all_files(args.data_dir)
        else:
            logger.info("No changed files specified - processing all files")
            result = processor.process_all_files(args.data_dir)
    
    # Print summary
    stats = processor.get_collection_stats()
    logger.info(f"Final collection stats: {stats}")
    
    # Output results for GitHub Actions
    with open('processing_results.json', 'w') as f:
        json.dump({
            'result': result,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Exit with error if processing failed
    if not result.get('success', True):
        sys.exit(1)

if __name__ == "__main__":
    main()