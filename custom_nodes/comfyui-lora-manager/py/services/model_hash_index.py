from typing import Dict, Optional, Set, List
import os

class ModelHashIndex:
    """Index for looking up models by hash or filename"""
    
    def __init__(self):
        self._hash_to_path: Dict[str, str] = {}
        self._filename_to_hash: Dict[str, str] = {}
        # New data structures for tracking duplicates
        self._duplicate_hashes: Dict[str, List[str]] = {}  # sha256 -> list of paths
        self._duplicate_filenames: Dict[str, List[str]] = {}  # filename -> list of paths
    
    def add_entry(self, sha256: str, file_path: str) -> None:
        """Add or update hash index entry"""
        if not sha256 or not file_path:
            return
            
        # Ensure hash is lowercase for consistency
        sha256 = sha256.lower()
        
        # Extract filename without extension
        filename = self._get_filename_from_path(file_path)
        
        # Track duplicates by hash
        if sha256 in self._hash_to_path:
            old_path = self._hash_to_path[sha256]
            if old_path != file_path:  # Only record if it's actually a different path
                if sha256 not in self._duplicate_hashes:
                    self._duplicate_hashes[sha256] = [old_path]
                if file_path not in self._duplicate_hashes.get(sha256, []):
                    self._duplicate_hashes.setdefault(sha256, []).append(file_path)
        
        # Track duplicates by filename - FIXED LOGIC
        if filename in self._filename_to_hash:
            existing_hash = self._filename_to_hash[filename]
            existing_path = self._hash_to_path.get(existing_hash)
            
            # If this is a different file with the same filename
            if existing_path and existing_path != file_path:
                # Initialize duplicates tracking if needed
                if filename not in self._duplicate_filenames:
                    self._duplicate_filenames[filename] = [existing_path]
                
                # Add current file to duplicates if not already present
                if file_path not in self._duplicate_filenames[filename]:
                    self._duplicate_filenames[filename].append(file_path)
        
        # Remove old path mapping if hash exists
        if sha256 in self._hash_to_path:
            old_path = self._hash_to_path[sha256]
            old_filename = self._get_filename_from_path(old_path)
            if old_filename in self._filename_to_hash and self._filename_to_hash[old_filename] == sha256:
                del self._filename_to_hash[old_filename]
        
        # Remove old hash mapping if filename exists and points to different hash
        if filename in self._filename_to_hash:
            old_hash = self._filename_to_hash[filename]
            if old_hash != sha256 and old_hash in self._hash_to_path:
                # Don't delete the old hash mapping, just update filename mapping
                pass
        
        # Add new mappings
        self._hash_to_path[sha256] = file_path
        self._filename_to_hash[filename] = sha256
    
    def _get_filename_from_path(self, file_path: str) -> str:
        """Extract filename without extension from path"""
        return os.path.splitext(os.path.basename(file_path))[0]
    
    def remove_by_path(self, file_path: str, hash_val: str = None) -> None:
        """Remove entry by file path"""
        filename = self._get_filename_from_path(file_path)
        
        # Find the hash for this file path
        if hash_val is None:
            for h, p in self._hash_to_path.items():
                if p == file_path:
                    hash_val = h
                    break
        
        # If we didn't find a hash, nothing to do
        if not hash_val:
            return
        
        # Update duplicates tracking for hash
        if hash_val in self._duplicate_hashes:
            # Remove the current path from duplicates
            self._duplicate_hashes[hash_val] = [p for p in self._duplicate_hashes[hash_val] if p != file_path]
            
            # Update or remove hash mapping based on remaining duplicates
            if len(self._duplicate_hashes[hash_val]) > 0:
                # Replace with one of the remaining paths
                new_path = self._duplicate_hashes[hash_val][0]
                new_filename = self._get_filename_from_path(new_path)
                
                # Update hash-to-path mapping
                self._hash_to_path[hash_val] = new_path
                
                # IMPORTANT: Update filename-to-hash mapping for consistency
                # Remove old filename mapping if it points to this hash
                if filename in self._filename_to_hash and self._filename_to_hash[filename] == hash_val:
                    del self._filename_to_hash[filename]
                
                # Add new filename mapping
                self._filename_to_hash[new_filename] = hash_val
                
                # If only one duplicate left, remove from duplicates tracking
                if len(self._duplicate_hashes[hash_val]) == 1:
                    del self._duplicate_hashes[hash_val]
            else:
                # No duplicates left, remove hash entry completely
                del self._duplicate_hashes[hash_val]
                del self._hash_to_path[hash_val]
                
                # Remove corresponding filename entry if it points to this hash
                if filename in self._filename_to_hash and self._filename_to_hash[filename] == hash_val:
                    del self._filename_to_hash[filename]
        else:
            # No duplicates, simply remove the hash entry
            del self._hash_to_path[hash_val]
            
            # Remove corresponding filename entry if it points to this hash
            if filename in self._filename_to_hash and self._filename_to_hash[filename] == hash_val:
                del self._filename_to_hash[filename]
        
        # Update duplicates tracking for filename
        if filename in self._duplicate_filenames:
            # Remove the current path from duplicates
            self._duplicate_filenames[filename] = [p for p in self._duplicate_filenames[filename] if p != file_path]
            
            # Update or remove filename mapping based on remaining duplicates
            if len(self._duplicate_filenames[filename]) > 0:
                # Get the hash for the first remaining duplicate path
                first_dup_path = self._duplicate_filenames[filename][0]
                first_dup_hash = None
                for h, p in self._hash_to_path.items():
                    if p == first_dup_path:
                        first_dup_hash = h
                        break
                
                # Update the filename to hash mapping if we found a hash
                if first_dup_hash:
                    self._filename_to_hash[filename] = first_dup_hash
                
                # If only one duplicate left, remove from duplicates tracking
                if len(self._duplicate_filenames[filename]) == 1:
                    del self._duplicate_filenames[filename]
            else:
                # No duplicates left, remove filename entry completely
                del self._duplicate_filenames[filename]
                if filename in self._filename_to_hash:
                    del self._filename_to_hash[filename]
    
    def remove_by_hash(self, sha256: str) -> None:
        """Remove entry by hash"""
        sha256 = sha256.lower()
        if sha256 not in self._hash_to_path:
            return
        
        # Get the path and filename
        path = self._hash_to_path[sha256]
        filename = self._get_filename_from_path(path)
        
        # Get all paths for this hash (including duplicates)
        paths_to_remove = [path]
        if sha256 in self._duplicate_hashes:
            paths_to_remove.extend(self._duplicate_hashes[sha256])
            del self._duplicate_hashes[sha256]
        
        # Remove hash-to-path mapping
        del self._hash_to_path[sha256]
        
        # Update filename-to-hash and duplicate filenames for all paths
        for path_to_remove in paths_to_remove:
            fname = self._get_filename_from_path(path_to_remove)
            
            # If this filename maps to the hash we're removing, remove it
            if fname in self._filename_to_hash and self._filename_to_hash[fname] == sha256:
                del self._filename_to_hash[fname]
            
            # Update duplicate filenames tracking
            if fname in self._duplicate_filenames:
                self._duplicate_filenames[fname] = [p for p in self._duplicate_filenames[fname] if p != path_to_remove]
                
                if not self._duplicate_filenames[fname]:
                    del self._duplicate_filenames[fname]
                elif len(self._duplicate_filenames[fname]) == 1:
                    # If only one entry remains, it's no longer a duplicate
                    del self._duplicate_filenames[fname]
    
    def has_hash(self, sha256: str) -> bool:
        """Check if hash exists in index"""
        return sha256.lower() in self._hash_to_path
    
    def get_path(self, sha256: str) -> Optional[str]:
        """Get file path for a hash"""
        return self._hash_to_path.get(sha256.lower())
    
    def get_hash(self, file_path: str) -> Optional[str]:
        """Get hash for a file path"""
        filename = self._get_filename_from_path(file_path)
        return self._filename_to_hash.get(filename)
    
    def get_hash_by_filename(self, filename: str) -> Optional[str]:
        """Get hash for a filename without extension"""
        return self._filename_to_hash.get(filename)
    
    def clear(self) -> None:
        """Clear all entries"""
        self._hash_to_path.clear()
        self._filename_to_hash.clear()
        self._duplicate_hashes.clear()
        self._duplicate_filenames.clear()
    
    def get_all_hashes(self) -> Set[str]:
        """Get all hashes in the index"""
        return set(self._hash_to_path.keys())
    
    def get_all_filenames(self) -> Set[str]:
        """Get all filenames in the index"""
        return set(self._filename_to_hash.keys())
    
    def get_duplicate_hashes(self) -> Dict[str, List[str]]:
        """Get dictionary of duplicate hashes and their paths"""    
        return self._duplicate_hashes
    
    def get_duplicate_filenames(self) -> Dict[str, List[str]]:
        """Get dictionary of duplicate filenames and their paths"""
        return self._duplicate_filenames
    
    def __len__(self) -> int:
        """Get number of entries"""
        return len(self._hash_to_path)