#!/usr/bin/env python3
"""
Vector processor with incremental update support.
Processes only changed files instead of reprocessing everything.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Import your vector DB library (adjust based on your setup)
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("⚠️ ChromaDB not installed. Install with: pip install chromadb")
    sys.exit(1)


class IncrementalVectorProcessor:
    def __init__(self, data_dir: str, chroma_persist_dir: str = "./chroma_db"):
        self.data_dir = Path(data_dir)
        self.chroma_persist_dir = Path(chroma_persist_dir)
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=str(self.chroma_persist_dir),
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="code_documents",
            metadata={"description": "Source code and documentation embeddings"}
        )
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "mode": "unknown",
            "files_processed": 0,
            "files_removed": 0,
            "files_skipped": 0,
            "errors": []
        }
    
    def process_full(self):
        """Process all files in the data directory."""
        print("🚀 Starting FULL processing mode")
        self.results["mode"] = "full"
        
        all_files = self._get_all_files()
        print(f"📊 Found {len(all_files)} files to process")
        
        for file_path in all_files:
            try:
                self._process_file(file_path)
                self.results["files_processed"] += 1
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                print(f"❌ {error_msg}")
                self.results["errors"].append(error_msg)
        
        self._save_results()
        print(f"✅ Full processing complete: {self.results['files_processed']} files")
    
    def process_incremental(self, file_list_path: str):
        """Process only files listed in the file list."""
        print("🎯 Starting INCREMENTAL processing mode")
        self.results["mode"] = "incremental"
        
        if not os.path.exists(file_list_path):
            print(f"❌ File list not found: {file_list_path}")
            self.results["errors"].append(f"File list not found: {file_list_path}")
            self._save_results()
            return
        
        with open(file_list_path, 'r') as f:
            files_to_process = [line.strip() for line in f if line.strip()]
        
        # Check if this is a full sync marker
        if files_to_process == ["FULL_SYNC"]:
            print("📦 Full sync detected, processing all files")
            self.process_full()
            return
        
        print(f"📊 Processing {len(files_to_process)} changed file(s)")
        
        for file_entry in files_to_process:
            try:
                if file_entry.startswith("REMOVED:"):
                    # Handle file removal
                    file_path = file_entry.replace("REMOVED:", "")
                    self._remove_file_from_db(file_path)
                    self.results["files_removed"] += 1
                else:
                    # Process new or modified file
                    file_path = Path(file_entry)
                    if file_path.exists():
                        self._process_file(file_path)
                        self.results["files_processed"] += 1
                    else:
                        print(f"⚠️ File not found: {file_path}")
                        self.results["files_skipped"] += 1
            except Exception as e:
                error_msg = f"Error processing {file_entry}: {str(e)}"
                print(f"❌ {error_msg}")
                self.results["errors"].append(error_msg)
        
        self._save_results()
        print(f"✅ Incremental processing complete:")
        print(f"   - Processed: {self.results['files_processed']} files")
        print(f"   - Removed: {self.results['files_removed']} files")
        print(f"   - Skipped: {self.results['files_skipped']} files")
    
    def process_test_mode(self, max_files: int = 10):
        """Process only the first N files for testing."""
        print(f"🧪 Starting TEST mode (processing {max_files} files)")
        self.results["mode"] = "test"
        
        all_files = self._get_all_files()
        test_files = all_files[:max_files]
        
        print(f"📊 Processing {len(test_files)} test files")
        
        for file_path in test_files:
            try:
                self._process_file(file_path)
                self.results["files_processed"] += 1
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                print(f"❌ {error_msg}")
                self.results["errors"].append(error_msg)
        
        self._save_results()
        print(f"✅ Test processing complete: {self.results['files_processed']} files")
    
    def _get_all_files(self) -> List[Path]:
        """Get all processable files in the data directory."""
        extensions = {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.rst', '.csv'}
        files = []
        
        for ext in extensions:
            files.extend(self.data_dir.rglob(f"*{ext}"))
        
        return sorted(files)
    
    def _process_file(self, file_path: Path):
        """Process a single file and add/update it in the vector database."""
        print(f"📄 Processing: {file_path}")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Generate a unique ID for this file
            file_id = str(file_path.relative_to(self.data_dir))
            
            # Create metadata
            metadata = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix,
                "relative_path": file_id,
                "size": len(content),
                "processed_at": datetime.now().isoformat()
            }
            
            # Remove existing entry if it exists (for updates)
            try:
                self.collection.delete(ids=[file_id])
            except:
                pass  # File didn't exist in DB yet
            
            # Add to collection
            # ChromaDB will automatically generate embeddings
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[file_id]
            )
            
            print(f"  ✓ Added to vector DB: {file_id}")
            
        except Exception as e:
            raise Exception(f"Failed to process file: {str(e)}")
    
    def _remove_file_from_db(self, file_path: str):
        """Remove a file from the vector database."""
        print(f"🗑️ Removing from DB: {file_path}")
        
        try:
            # Extract relative path for ID
            path = Path(file_path)
            if path.is_absolute():
                file_id = str(path.relative_to(self.data_dir))
            else:
                file_id = str(path)
            
            # Delete from collection
            self.collection.delete(ids=[file_id])
            print(f"  ✓ Removed from vector DB: {file_id}")
            
        except Exception as e:
            print(f"  ⚠️ Could not remove {file_path}: {str(e)}")
    
    def _save_results(self):
        """Save processing results to JSON file."""
        with open("processing_results.json", "w") as f:
            json.dump(self.results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Process files into vector database")
    parser.add_argument("--data-dir", required=True, help="Directory containing data files")
    parser.add_argument("--chroma-dir", default="./chroma_db", help="ChromaDB persistence directory")
    parser.add_argument("--test-mode", action="store_true", help="Process only 10 files for testing")
    parser.add_argument("--incremental", action="store_true", help="Process only changed files")
    parser.add_argument("--file-list", help="Path to file containing list of files to process")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = IncrementalVectorProcessor(args.data_dir, args.chroma_dir)
    
    # Choose processing mode
    if args.test_mode:
        processor.process_test_mode()
    elif args.incremental and args.file_list:
        processor.process_incremental(args.file_list)
    else:
        processor.process_full()


if __name__ == "__main__":
    main()