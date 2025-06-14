

import git 
import json
import os
import logging
import httpx 
from pathlib import Path
from typing import List, Dict, Any, Optional 
from pydantic import BaseModel # BaseModel is used as a type hint for item_model_class
from fastapi import HTTPException, status
import shutil
from git import exc as git_exc

from app.models.github_models import QuestionQueryPairItem, SchemaItem 

# If github_models.py is a separate file, you would uncomment these:
from app.models.github_models import  GitRepoTargetConfig 

GITHUB_API_URL = "https://api.github.com"
GIT_SERVER_BASE_CLONE_PATH = os.getenv("GIT_SERVER_BASE_CLONE_PATH", "/tmp/app_git_clones")


class GitHubService:
    def __init__(self, 
                 author_name: str = "RAG Service Bot", 
                 author_email: str = "bot@rag.service.com"):
        self.author = git.Actor(author_name, author_email)
        Path(GIT_SERVER_BASE_CLONE_PATH).mkdir(parents=True, exist_ok=True)
        logging.info(f"GitHubService initialized. Base clone path: {GIT_SERVER_BASE_CLONE_PATH}")


    def _get_local_repo_path(self, repo_url: str) -> Path:
        try:
            repo_name_part = repo_url.split('/')[-1].replace('.git', '')
            user_or_org_part = repo_url.split('/')[-2]
        except IndexError:
            logging.error(f"Invalid repo_url format for path generation: {repo_url}")
            raise ValueError("Invalid repo_url format for generating local path.")
        safe_user_org = "".join(c if c.isalnum() else "_" for c in user_or_org_part)
        safe_repo_name = "".join(c if c.isalnum() else "_" for c in repo_name_part)
        return Path(GIT_SERVER_BASE_CLONE_PATH) / f"{safe_user_org}_{safe_repo_name}"

    def _construct_authenticated_url(self, repo_url: str, token: str) -> str:
        if not token: raise ValueError("GitHub token is required for authenticated URL.")
        try:
            protocol, rest_of_url = repo_url.split("://", 1)
            return f"{protocol}://oauth2:{token}@{rest_of_url}"
        except ValueError:
            logging.error(f"Invalid repository URL format: {repo_url}")
            raise ValueError("Invalid repository URL format.")

    # def _ensure_repo_is_ready(self, repo_url: str, branch: str, token: str, local_repo_path: Path) -> git.Repo:
    #     authenticated_repo_url = self._construct_authenticated_url(repo_url, token)
    #     repo_instance: Optional[git.Repo] = None
    #     try:
    #         if local_repo_path.exists() and local_repo_path.joinpath(".git").exists():
    #             logging.info(f"Opening existing repository at {local_repo_path}")
    #             repo_instance = git.Repo(str(local_repo_path))
    #             if repo_instance.bare:
    #                  logging.error(f"Repository at {local_repo_path} is bare. Re-cloning.")
    #                  shutil.rmtree(local_repo_path) 
    #                  repo_instance = self._clone_repo_to_path(authenticated_repo_url, local_repo_path, branch)
    #             else:
    #                 self._checkout_and_pull_specific(repo_instance, branch, authenticated_repo_url)
    #         else:
    #             logging.info(f"Local repository not found at {local_repo_path}. Cloning new.")
    #             repo_instance = self._clone_repo_to_path(authenticated_repo_url, local_repo_path, branch)
    #     except git.exc.NoSuchPathError: 
    #         logging.warning(f"Path {local_repo_path} exists but not a valid Git repo. Attempting to re-clone.")
    #         if local_repo_path.exists(): 
    #             if local_repo_path.is_dir(): shutil.rmtree(local_repo_path)
    #             else: local_repo_path.unlink()
    #         repo_instance = self._clone_repo_to_path(authenticated_repo_url, local_repo_path, branch)
    #     except Exception as e:
    #         logging.error(f"Critical error ensuring repository readiness for {repo_url} at {local_repo_path}: {e}")
    #         raise RuntimeError(f"Failed to initialize or access repository: {e}")
        
    #     if not repo_instance: 
    #         raise RuntimeError(f"Failed to obtain repository instance for {local_repo_path}")
    #     return repo_instance
    def _ensure_repo_is_ready(
            self,
            repo_url: str,
            branch: str,
            token: str,
            local_repo_path: Path
    ) -> git.Repo:
        """Return a ready-to-use Repo object, cloning or repairing if necessary."""
        authenticated_repo_url = self._construct_authenticated_url(repo_url, token)

        try:
            # ── 1. Directory already holds a repo ───────────────────────────
            if local_repo_path.joinpath(".git").is_dir():
                logging.info("Opening existing repository at %s", local_repo_path)
                repo = git.Repo(str(local_repo_path))

                # Repo object exists but is bare ⇒ nuke & reclone
                if repo.bare:
                    logging.warning("Repository is bare, recloning.")
                    shutil.rmtree(local_repo_path, ignore_errors=True)
                    repo = self._clone_repo_to_path(
                        authenticated_repo_url, local_repo_path, branch
                    )
                else:
                    self._checkout_and_pull_specific(repo, branch, authenticated_repo_url)

            # ── 2. No repo on disk yet ──────────────────────────────────────
            else:
                logging.info("Local repository not found. Cloning fresh copy.")
                repo = self._clone_repo_to_path(
                    authenticated_repo_url, local_repo_path, branch
                )

        # ── 3. The directory exists but isn’t a valid repo ───────────────────
        except (git_exc.NoSuchPathError, git_exc.InvalidGitRepositoryError):
            logging.warning("Path exists but is not a valid repo – recloning.")
            shutil.rmtree(local_repo_path, ignore_errors=True)
            repo = self._clone_repo_to_path(
                authenticated_repo_url, local_repo_path, branch
            )

        # ── 4. Low-level Git failures (network, auth, etc.) ──────────────────
        except git_exc.GitCommandError as e:
            logging.error("Git command failed: %s", e.stderr or e)
            raise RuntimeError(f"Git operation failed: {e.stderr or e}") from e

        # ── 5. Anything else ────────────────────────────────────────────────
        except Exception as e:
            logging.error(
                "Critical error ensuring repository readiness for %s at %s: %s",
                repo_url, local_repo_path, e
            )
            raise RuntimeError(f"Failed to initialize or access repository: {e}") from e

        # Final sanity check
        if repo is None:
            raise RuntimeError(f"Failed to obtain repository instance for {local_repo_path}")

        return repo

    def _clone_repo_to_path(self, authenticated_repo_url: str, local_repo_path: Path, branch: str) -> git.Repo:
        logging.info(f"Cloning repository to {local_repo_path} (branch: {branch})")
        local_repo_path.mkdir(parents=True, exist_ok=True) 
        cloned_repo: git.Repo
        try:
            try: 
                cloned_repo = git.Repo.clone_from(authenticated_repo_url, str(local_repo_path), branch=branch)
            except git.exc.GitCommandError as e_branch:
                if "Remote branch" in str(e_branch.stderr) and "not found" in str(e_branch.stderr):
                    logging.warning(f"Branch '{branch}' not found on remote during initial clone. Cloning default branch and creating '{branch}' locally.")
                    cloned_repo = git.Repo.clone_from(authenticated_repo_url, str(local_repo_path))
                    if branch not in [b.name for b in cloned_repo.branches]:
                        cloned_repo.create_head(branch)
                    cloned_repo.git.checkout(branch)
                else: raise 
            logging.info(f"Successfully cloned repository to {local_repo_path}")
            return cloned_repo
        except git.exc.GitCommandError as e:
            logging.error(f"Git clone failed: {e.stderr}. URL: {authenticated_repo_url[:30]}...")
            raise RuntimeError(f"Git clone failed. Check repository URL, branch, and token permissions. Error: {e.stderr}")
        except Exception as e: 
            logging.error(f"Unexpected error during clone to {local_repo_path}: {e}")
            raise RuntimeError(f"Unexpected error during repository clone: {e}")


    def _checkout_and_pull_specific(self, repo: git.Repo, branch: str, authenticated_repo_url: str):
        try:
            if branch not in [b.name for b in repo.branches]:
                origin = repo.remotes.origin
                with origin.config_writer as cw: cw.set("url", authenticated_repo_url)
                origin.fetch() 
                if f"origin/{branch}" in [r.name for r in repo.remotes.origin.refs]:
                    repo.create_head(branch, repo.remotes.origin.refs[branch]).set_tracking_branch(repo.remotes.origin.refs[branch])
                    logging.info(f"Created local branch '{branch}' tracking 'origin/{branch}'.")
                else:
                    logging.info(f"Branch '{branch}' not found on remote 'origin'. Creating new local branch '{branch}'.")
                    repo.create_head(branch)
            
            if repo.active_branch.name != branch:
                logging.info(f"Current branch is {repo.active_branch.name}, switching to configured branch '{branch}'")
                repo.git.checkout(branch)
            
            origin = repo.remotes.origin
            with origin.config_writer as cw: cw.set("url", authenticated_repo_url) 
            logging.info(f"Attempting to pull from {origin.url[:30]}... on branch {repo.active_branch.name}")
            
            stashed = False
            if repo.is_dirty():
                logging.warning(f"Local changes detected in {repo.working_dir} on branch {branch}. Stashing before merge.")
                repo.git.stash()
                stashed = True

            if f"origin/{branch}" in [r.name for r in repo.remotes.origin.refs]: 
                repo.git.merge(f"origin/{branch}") 
                logging.info("Pull (fetch & merge) successful or up-to-date.")
            else:
                logging.info(f"Remote branch origin/{branch} not found after fetch. Assuming new branch or local-only work.")

            if stashed:
                try: repo.git.stash("pop") 
                except git.exc.GitCommandError as stash_err: logging.warning(f"Could not pop stash after merge (maybe no stash or conflicts): {stash_err.stderr}")

        except git.exc.GitCommandError as e: 
            logging.error(f"Git pull/merge command failed: {e.stderr}")
            logging.warning(f"Proceeding with local version due to pull/merge failure. Push might fail if outdated or due to conflicts.")
        except Exception as e: 
            logging.error(f"Unexpected error during git checkout/pull: {e}")
            logging.warning(f"Unexpected error during git checkout/pull: {e}. Proceeding with local version.")


    def _save_data_to_file_and_commit(self, repo: git.Repo, local_repo_path: Path, file_path_in_repo_str: str, data_to_save: List[BaseModel], commit_message: str, branch: str, authenticated_repo_url: str) -> str:
        full_file_path = local_repo_path / file_path_in_repo_str
        try:
            full_file_path.parent.mkdir(parents=True, exist_ok=True) 
            with open(full_file_path, "w") as f: 
                json.dump([item.model_dump(exclude_none=True) for item in data_to_save], f, indent=2)
            
            repo.index.add([str(full_file_path)])
            
            if not repo.index.diff("HEAD") and not repo.is_dirty(untracked_files=True): # Check staged & working tree (for untracked)
                 logging.info(f"No changes to commit to {file_path_in_repo_str}.")
                 return repo.head.commit.hexsha if repo.head.is_valid() else "No new commit (no changes staged)"

            commit = repo.index.commit(commit_message, author=self.author, committer=self.author)
            logging.info(f"Committed changes to {file_path_in_repo_str} locally: {commit.hexsha}")
            
            origin = repo.remotes.origin
            with origin.config_writer as cw: cw.set("url", authenticated_repo_url)
            
            logging.info(f"Pushing branch '{branch}' to remote 'origin'")
            push_info_list = origin.push(refspec=f"{branch}:{branch}")
            
            for push_info in push_info_list:
                if push_info.flags & git.PushInfo.ERROR: raise RuntimeError(f"Failed to push to GitHub: {push_info.summary or str(push_info.error if hasattr(push_info, 'error') else 'Unknown push error')}")
                elif push_info.flags & git.PushInfo.REJECTED: raise RuntimeError(f"Push to GitHub rejected: {push_info.summary}")

            logging.info(f"Successfully pushed {file_path_in_repo_str} changes to branch '{branch}'")
            return commit.hexsha
        except Exception as e:
            logging.error(f"Error in _save_data_to_file_and_commit for {file_path_in_repo_str}: {e}")
            raise RuntimeError(f"Git operation failed for {file_path_in_repo_str}: {e}")

    def _load_data_from_file(self, repo: git.Repo, local_repo_path: Path, file_path_in_repo_str: str, item_model_class: type[BaseModel]) -> Dict[str, Any]:
        full_file_path = local_repo_path / file_path_in_repo_str
        try:
            if not full_file_path.exists():
                return {"data": [], "error": f"File '{file_path_in_repo_str}' not found."}
            with open(full_file_path, "r") as f: raw_data = json.load(f)
            if not isinstance(raw_data, list): raise RuntimeError(f"Invalid data format in {file_path_in_repo_str}: expected a list.")
            # Ensure SchemaItem and QuestionQueryPairItem are imported or passed if this is a generic helper
            parsed_data = [item_model_class(**item) for item in raw_data] 
            commit_hash = repo.head.commit.hexsha if repo.head.is_valid() else None
            commit_date = repo.head.commit.committed_datetime.isoformat() if repo.head.is_valid() else None
            return {"data": parsed_data, "commit_hash": commit_hash, "last_modified": commit_date}
        except Exception as e: 
            raise RuntimeError(f"Error loading/parsing {file_path_in_repo_str}: {e}")

    def save_schema_data_to_repo(self, git_config: 'GitRepoTargetConfig', target_file_name: str, schema_items: List['SchemaItem'], commit_message: str) -> Dict[str, Any]:
        local_repo_path = self._get_local_repo_path(git_config.repo_url)
        repo = self._ensure_repo_is_ready(git_config.repo_url, git_config.branch, git_config.github_pat, local_repo_path)
        authenticated_url = self._construct_authenticated_url(git_config.repo_url, git_config.github_pat)
        commit_hash = self._save_data_to_file_and_commit(repo, local_repo_path, target_file_name, schema_items, commit_message, git_config.branch, authenticated_url)
        return {"message": "Schema data saved to specified GitHub repository.", "commit_hash": commit_hash}
    
    def load_schema_data_from_repo(self, git_config: 'GitRepoTargetConfig', target_file_name: str) -> Dict[str, Any]:
        local_repo_path = self._get_local_repo_path(git_config.repo_url)
        repo = self._ensure_repo_is_ready(git_config.repo_url, git_config.branch, git_config.github_pat, local_repo_path)
        result = self._load_data_from_file(repo, local_repo_path, target_file_name, SchemaItem) # Pass SchemaItem model

        # Remap to the exact shape the response model expects:
        response = {
            "schema_data": result.get("schema_data") or result.get("data"),
            "target_file_name": target_file_name,  
            "commit_hash": result.get("commit_hash"),
            "last_modified": result.get("last_modified"),
            "message": result.get("error") or f"Schema data loaded successfully from {git_config.repo_url} (branch '{git_config.branch}').",
            "error": result.get("error"),
        }
        return response

    def save_qq_data_to_repo(self, git_config: 'GitRepoTargetConfig', target_file_name: str, qq_items: List['QuestionQueryPairItem'], commit_message: str) -> Dict[str, Any]:
        local_repo_path = self._get_local_repo_path(git_config.repo_url)
        repo = self._ensure_repo_is_ready(git_config.repo_url, git_config.branch, git_config.github_pat, local_repo_path)
        authenticated_url = self._construct_authenticated_url(git_config.repo_url, git_config.github_pat)
        commit_hash = self._save_data_to_file_and_commit(repo, local_repo_path, target_file_name, qq_items, commit_message, git_config.branch, authenticated_url)
        return {"message": "Q/Q data saved to specified GitHub repository.", "commit_hash": commit_hash}

    # def load_qq_data_from_repo(self, git_config: 'GitRepoTargetConfig', target_file_name: str) -> Dict[str, Any]:
    #     local_repo_path = self._get_local_repo_path(git_config.repo_url)
    #     repo = self._ensure_repo_is_ready(git_config.repo_url, git_config.branch, git_config.github_pat, local_repo_path)
    #     result = self._load_data_from_file(repo, local_repo_path, target_file_name, QuestionQueryPairItem) # Pass QuestionQueryPairItem model
    #     return {**result, "message": result.get("error") or f"Q/Q data loaded successfully from {git_config.repo_url} (branch '{git_config.branch}')."}
    def load_qq_data_from_repo(self, git_config: 'GitRepoTargetConfig', target_file_name: str) -> Dict[str, Any]:
        """
        Loads Question/Query pair data from a specified file in a GitHub repository.
        This method's response structure is designed to align with LoadQQFromGitResponse Pydantic model.
        """
        local_repo_path = self._get_local_repo_path(git_config.repo_url)
        repo = self._ensure_repo_is_ready(git_config.repo_url, git_config.branch, git_config.github_pat, local_repo_path)
        
        # Call the generic _load_data_from_file method, passing the QuestionQueryPairItem model
        # for parsing the file content.
        # _load_data_from_file is assumed to return a dictionary like:
        # {
        #   'data': List[QuestionQueryPairItem] (or None),
        #   'commit_hash': str (optional),
        #   'last_modified': str (optional),
        #   'error': str (optional, if an error occurred reading/parsing the file)
        # }
        result = self._load_data_from_file(
            repo, 
            local_repo_path, 
            target_file_name, 
            QuestionQueryPairItem  # Specific model for Q/Q items
        )

        loaded_items = result.get("data")
        final_qq_data: List[QuestionQueryPairItem] = [] # Ensure it's always a list

        # Populate final_qq_data only if no error occurred during file loading AND data is not None.
        # The corresponding Pydantic response model (LoadQQFromGitResponse) likely has:
        # `qq_data: List[QuestionQueryPairItem]`, so it must be a list, not None.
        if result.get("error") is None and loaded_items is not None:
            final_qq_data = loaded_items
        # If result.get("error") is not None, final_qq_data remains [].
        # If result.get("error") is None but loaded_items (result.get("data")) is None,
        # it means no data was parsed (e.g., empty file), so final_qq_data correctly remains [].
        
        # Construct the response dictionary to precisely match the fields expected by
        # the LoadQQFromGitResponse Pydantic model.
        response_payload = {
            "qq_data": final_qq_data,
            "commit_hash": result.get("commit_hash"),
            "last_modified": result.get("last_modified"),
            "message": result.get("error") or f"Q/Q data loaded successfully from {git_config.repo_url} (branch '{git_config.branch}').",
            "error": result.get("error"),
            # Note: The 'target_file_name' field is included in the load_schema_data_from_repo response construction.
            # If you want it in the Q/Q response as well, you would add it here AND ensure
            # your LoadQQFromGitResponse Pydantic model includes a 'target_file_name' field.
            # For now, it's omitted to strictly match a typical LoadQQFromGitResponse model that might not have it.
        }
        
        return response_payload
    
    async def create_repository_on_github(self, user_provided_pat: str, repo_name: str, description: Optional[str], private: bool) -> Dict[str, Any]:
        if not user_provided_pat: 
            raise ValueError("A GitHub Personal Access Token is required to create repositories.")
        headers = {"Authorization": f"token {user_provided_pat}", "Accept": "application/vnd.github.v3+json", "X-GitHub-Api-Version": "2022-11-28"}
        payload = {"name": repo_name, "description": description or "", "private": private, "auto_init": True}
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{GITHUB_API_URL}/user/repos", headers=headers, json=payload)
                response.raise_for_status() 
                repo_data = response.json()
                return {
                    "name": repo_data.get("name"), "html_url": repo_data.get("html_url"),
                    "ssh_url": repo_data.get("ssh_url"), "clone_url": repo_data.get("clone_url"),
                    "message": f"Successfully created GitHub repository '{repo_data.get('full_name')}'."
                }
            except httpx.HTTPStatusError as e:
                error_detail = f"GitHub API error creating repo: {e.response.status_code} - {e.response.text}"
                logging.error(error_detail)
                raise RuntimeError(error_detail) 
            except Exception as e:
                logging.error(f"Unexpected error creating GitHub repository: {e}")
                raise RuntimeError(f"Unexpected error creating GitHub repository: {e}")

# --- Dependency to provide GitHubService instance ---
def get_github_service_dependency() -> GitHubService:
    try:
        author_name = os.getenv("GIT_COMMIT_AUTHOR_NAME", "RAG Service Bot")
        author_email = os.getenv("GIT_COMMIT_AUTHOR_EMAIL", "bot@rag.service.com") 
            
        # Service is instantiated with only author details.
        # Token and repo specifics are passed per-call to relevant methods.
        return GitHubService(
            author_name=author_name,
            author_email=author_email
        )
    except Exception as e: 
        logging.error(f"Fatal error creating GitHubService instance: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not initialize GitHub service: {str(e)}")
