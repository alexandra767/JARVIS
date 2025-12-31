#!/usr/bin/env python3
"""
JARVIS File Operations - Create, Read, Edit Documents and Files
Gives Jarvis the ability to work with files on your computer
"""

import os
import asyncio
import aiofiles
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import shutil
import mimetypes

logger = logging.getLogger("JarvisFiles")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Allowed directories for file operations (security)
ALLOWED_DIRECTORIES = [
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Downloads"),
    os.path.expanduser("~/Projects"),
    os.path.expanduser("~/ai-clone-chat/jarvis_files"),
    os.path.expanduser("~/ai-clone-chat/jarvis_workspace"),
    "/home/alexandratitus767/ai-clone-chat",
]

# Jarvis workspace directory
JARVIS_WORKSPACE = "/home/alexandratitus767/ai-clone-chat/jarvis_workspace"

# Ensure workspace exists
os.makedirs(JARVIS_WORKSPACE, exist_ok=True)


# ============================================================================
# FILE OPERATIONS
# ============================================================================

class JarvisFileManager:
    """
    File management capabilities for Jarvis.
    Can create, read, edit, and organize files.
    """

    def __init__(self, workspace: str = None):
        self.workspace = workspace or JARVIS_WORKSPACE
        os.makedirs(self.workspace, exist_ok=True)

        # File type handlers
        self.text_extensions = {'.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml',
                                '.html', '.css', '.csv', '.xml', '.sh', '.bat'}
        self.document_extensions = {'.pdf', '.doc', '.docx', '.odt', '.rtf'}
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}

    def _is_safe_path(self, path: str) -> bool:
        """Check if path is within allowed directories"""
        abs_path = os.path.abspath(os.path.expanduser(path))

        # Always allow workspace
        if abs_path.startswith(os.path.abspath(self.workspace)):
            return True

        # Check allowed directories
        for allowed in ALLOWED_DIRECTORIES:
            if abs_path.startswith(os.path.abspath(allowed)):
                return True

        return False

    def _get_safe_path(self, filename: str, directory: str = None) -> str:
        """Get a safe path for a file"""
        if directory:
            if not self._is_safe_path(directory):
                raise PermissionError(f"Directory not allowed: {directory}")
            return os.path.join(directory, filename)
        return os.path.join(self.workspace, filename)

    # ==================== CREATE OPERATIONS ====================

    async def create_text_file(self, filename: str, content: str, directory: str = None) -> Dict[str, Any]:
        """Create a new text file"""
        try:
            path = self._get_safe_path(filename, directory)

            # Ensure directory exists
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else self.workspace, exist_ok=True)

            async with aiofiles.open(path, 'w') as f:
                await f.write(content)

            return {
                "success": True,
                "path": path,
                "filename": filename,
                "size": len(content),
                "message": f"Created file: {path}"
            }

        except Exception as e:
            logger.error(f"Create file error: {e}")
            return {"success": False, "error": str(e)}

    async def create_document(self, filename: str, content: str, doc_type: str = "markdown") -> Dict[str, Any]:
        """Create a document (markdown, text, etc.)"""
        if doc_type == "markdown":
            if not filename.endswith('.md'):
                filename += '.md'
        elif doc_type == "text":
            if not filename.endswith('.txt'):
                filename += '.txt'

        return await self.create_text_file(filename, content)

    async def create_note(self, title: str, content: str) -> Dict[str, Any]:
        """Create a quick note with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"note_{timestamp}_{title.replace(' ', '_')[:30]}.md"

        note_content = f"""# {title}

**Created:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

{content}
"""
        return await self.create_text_file(filename, note_content)

    async def create_project_folder(self, name: str, template: str = "basic") -> Dict[str, Any]:
        """Create a project folder with structure"""
        try:
            project_path = os.path.join(self.workspace, "projects", name)
            os.makedirs(project_path, exist_ok=True)

            # Create structure based on template
            if template == "basic":
                folders = ["docs", "src", "tests", "data"]
                files = {
                    "README.md": f"# {name}\n\nProject created by Jarvis on {datetime.now().strftime('%Y-%m-%d')}",
                    "notes.md": "# Project Notes\n\n",
                }
            elif template == "python":
                folders = ["src", "tests", "docs", "data", "notebooks"]
                files = {
                    "README.md": f"# {name}\n\n## Setup\n\n```bash\npip install -r requirements.txt\n```",
                    "requirements.txt": "# Add your dependencies here\n",
                    "src/__init__.py": "",
                    "tests/__init__.py": "",
                }
            else:
                folders = []
                files = {"README.md": f"# {name}\n"}

            # Create folders
            for folder in folders:
                os.makedirs(os.path.join(project_path, folder), exist_ok=True)

            # Create files
            for filename, content in files.items():
                file_path = os.path.join(project_path, filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(content)

            return {
                "success": True,
                "path": project_path,
                "folders": folders,
                "files": list(files.keys()),
                "message": f"Created project: {name}"
            }

        except Exception as e:
            logger.error(f"Create project error: {e}")
            return {"success": False, "error": str(e)}

    # ==================== READ OPERATIONS ====================

    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file's contents"""
        try:
            path = os.path.expanduser(path)

            if not self._is_safe_path(path):
                return {"success": False, "error": "Path not allowed"}

            if not os.path.exists(path):
                return {"success": False, "error": "File not found"}

            # Check file type
            ext = os.path.splitext(path)[1].lower()

            if ext in self.text_extensions:
                async with aiofiles.open(path, 'r') as f:
                    content = await f.read()
                return {
                    "success": True,
                    "path": path,
                    "content": content,
                    "size": len(content),
                    "type": "text"
                }
            else:
                # For binary files, return metadata only
                stat = os.stat(path)
                return {
                    "success": True,
                    "path": path,
                    "type": "binary",
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "message": "Binary file - content not displayed"
                }

        except Exception as e:
            logger.error(f"Read file error: {e}")
            return {"success": False, "error": str(e)}

    async def list_directory(self, path: str = None) -> Dict[str, Any]:
        """List contents of a directory"""
        try:
            path = os.path.expanduser(path) if path else self.workspace

            if not self._is_safe_path(path):
                return {"success": False, "error": "Path not allowed"}

            if not os.path.isdir(path):
                return {"success": False, "error": "Not a directory"}

            items = []
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                stat = os.stat(item_path)
                items.append({
                    "name": item,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

            # Sort: directories first, then files
            items.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))

            return {
                "success": True,
                "path": path,
                "items": items,
                "count": len(items)
            }

        except Exception as e:
            logger.error(f"List directory error: {e}")
            return {"success": False, "error": str(e)}

    async def search_files(self, query: str, path: str = None, extension: str = None) -> Dict[str, Any]:
        """Search for files by name or content"""
        try:
            search_path = os.path.expanduser(path) if path else self.workspace

            if not self._is_safe_path(search_path):
                return {"success": False, "error": "Path not allowed"}

            results = []
            query_lower = query.lower()

            for root, dirs, files in os.walk(search_path):
                for file in files:
                    # Filter by extension if specified
                    if extension and not file.endswith(extension):
                        continue

                    # Check filename match
                    if query_lower in file.lower():
                        file_path = os.path.join(root, file)
                        results.append({
                            "path": file_path,
                            "name": file,
                            "match_type": "filename"
                        })
                        continue

                    # Check content match for text files
                    ext = os.path.splitext(file)[1].lower()
                    if ext in self.text_extensions:
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', errors='ignore') as f:
                                content = f.read()
                                if query_lower in content.lower():
                                    results.append({
                                        "path": file_path,
                                        "name": file,
                                        "match_type": "content"
                                    })
                        except:
                            pass

                    # Limit results
                    if len(results) >= 50:
                        break

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            logger.error(f"Search files error: {e}")
            return {"success": False, "error": str(e)}

    # ==================== EDIT OPERATIONS ====================

    async def edit_file(self, path: str, content: str) -> Dict[str, Any]:
        """Replace file contents"""
        try:
            path = os.path.expanduser(path)

            if not self._is_safe_path(path):
                return {"success": False, "error": "Path not allowed"}

            # Backup original
            if os.path.exists(path):
                backup_path = path + ".backup"
                shutil.copy2(path, backup_path)

            async with aiofiles.open(path, 'w') as f:
                await f.write(content)

            return {
                "success": True,
                "path": path,
                "message": f"File updated: {path}"
            }

        except Exception as e:
            logger.error(f"Edit file error: {e}")
            return {"success": False, "error": str(e)}

    async def append_to_file(self, path: str, content: str) -> Dict[str, Any]:
        """Append content to a file"""
        try:
            path = os.path.expanduser(path)

            if not self._is_safe_path(path):
                return {"success": False, "error": "Path not allowed"}

            async with aiofiles.open(path, 'a') as f:
                await f.write(content)

            return {
                "success": True,
                "path": path,
                "message": f"Content appended to: {path}"
            }

        except Exception as e:
            logger.error(f"Append file error: {e}")
            return {"success": False, "error": str(e)}

    # ==================== DELETE/MOVE OPERATIONS ====================

    async def delete_file(self, path: str, confirm: bool = False) -> Dict[str, Any]:
        """Delete a file (requires confirmation)"""
        if not confirm:
            return {
                "success": False,
                "requires_confirmation": True,
                "message": f"Please confirm deletion of: {path}"
            }

        try:
            path = os.path.expanduser(path)

            if not self._is_safe_path(path):
                return {"success": False, "error": "Path not allowed"}

            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

            return {
                "success": True,
                "message": f"Deleted: {path}"
            }

        except Exception as e:
            logger.error(f"Delete file error: {e}")
            return {"success": False, "error": str(e)}

    async def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Move or rename a file"""
        try:
            source = os.path.expanduser(source)
            destination = os.path.expanduser(destination)

            if not self._is_safe_path(source) or not self._is_safe_path(destination):
                return {"success": False, "error": "Path not allowed"}

            shutil.move(source, destination)

            return {
                "success": True,
                "source": source,
                "destination": destination,
                "message": f"Moved to: {destination}"
            }

        except Exception as e:
            logger.error(f"Move file error: {e}")
            return {"success": False, "error": str(e)}

    async def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy a file"""
        try:
            source = os.path.expanduser(source)
            destination = os.path.expanduser(destination)

            if not self._is_safe_path(source) or not self._is_safe_path(destination):
                return {"success": False, "error": "Path not allowed"}

            if os.path.isdir(source):
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

            return {
                "success": True,
                "source": source,
                "destination": destination,
                "message": f"Copied to: {destination}"
            }

        except Exception as e:
            logger.error(f"Copy file error: {e}")
            return {"success": False, "error": str(e)}

    # ==================== SPECIAL OPERATIONS ====================

    async def open_file(self, path: str) -> Dict[str, Any]:
        """Open a file with the default application"""
        try:
            path = os.path.expanduser(path)

            if not self._is_safe_path(path):
                return {"success": False, "error": "Path not allowed"}

            if not os.path.exists(path):
                return {"success": False, "error": "File not found"}

            # Open with default application
            if os.name == 'nt':  # Windows
                os.startfile(path)
            elif os.name == 'posix':  # Linux/Mac
                subprocess.run(['xdg-open', path], check=True)

            return {
                "success": True,
                "path": path,
                "message": f"Opened: {path}"
            }

        except Exception as e:
            logger.error(f"Open file error: {e}")
            return {"success": False, "error": str(e)}

    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get detailed file information"""
        try:
            path = os.path.expanduser(path)

            if not os.path.exists(path):
                return {"success": False, "error": "File not found"}

            stat = os.stat(path)
            mime_type, _ = mimetypes.guess_type(path)

            return {
                "success": True,
                "path": path,
                "name": os.path.basename(path),
                "directory": os.path.dirname(path),
                "size": stat.st_size,
                "size_human": self._format_size(stat.st_size),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "mime_type": mime_type,
                "is_directory": os.path.isdir(path),
                "extension": os.path.splitext(path)[1]
            }

        except Exception as e:
            logger.error(f"File info error: {e}")
            return {"success": False, "error": str(e)}

    def _format_size(self, size: int) -> str:
        """Format file size to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


# ============================================================================
# TOOL DEFINITIONS FOR LLM
# ============================================================================

FILE_TOOL_DEFINITIONS = [
    {
        "name": "create_file",
        "description": "Create a new text file or document",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name of the file to create"},
                "content": {"type": "string", "description": "Content to write to the file"},
                "directory": {"type": "string", "description": "Optional directory path"}
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "edit_file",
        "description": "Edit/update a file's contents",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "New content for the file"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (optional, defaults to workspace)"}
            }
        }
    },
    {
        "name": "search_files",
        "description": "Search for files by name or content",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "path": {"type": "string", "description": "Directory to search in"},
                "extension": {"type": "string", "description": "Filter by file extension"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "create_note",
        "description": "Create a quick note with timestamp",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Note title"},
                "content": {"type": "string", "description": "Note content"}
            },
            "required": ["title", "content"]
        }
    },
    {
        "name": "create_project",
        "description": "Create a new project folder with structure",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Project name"},
                "template": {"type": "string", "enum": ["basic", "python"], "description": "Project template"}
            },
            "required": ["name"]
        }
    },
    {
        "name": "open_file",
        "description": "Open a file with the default application",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["path"]
        }
    }
]


# ============================================================================
# TEST
# ============================================================================

async def test_file_manager():
    """Test file operations"""
    fm = JarvisFileManager()

    print("Testing file operations...")

    # Create a note
    result = await fm.create_note("Test Note", "This is a test note created by Jarvis.")
    print(f"Create note: {result}")

    # List workspace
    result = await fm.list_directory()
    print(f"List workspace: {result}")

    # Read file
    if result.get("items"):
        first_file = result["items"][0]
        if first_file["type"] == "file":
            read_result = await fm.read_file(os.path.join(fm.workspace, first_file["name"]))
            print(f"Read file: {read_result}")

    # Create project
    result = await fm.create_project_folder("test_project", "python")
    print(f"Create project: {result}")

    # Search
    result = await fm.search_files("test")
    print(f"Search results: {result}")


if __name__ == "__main__":
    asyncio.run(test_file_manager())
