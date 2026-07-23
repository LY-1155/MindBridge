"""
Gap #9: .env 不入 git + 不入 Docker 镜像

验证行为：
  1. .env 不被 git 追踪
  2. .env.example 被 git 追踪（作为模板）
  3. .dockerignore 存在且排除 .env
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKERIGNORE = os.path.join(ROOT, ".dockerignore")


def _git_ls_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached"],
        cwd=ROOT, capture_output=True, text=True, timeout=10, check=True,
    )
    return result.stdout.strip().split("\n")


# ---------------------------------------------------------------------------
# 1. .env 不被 git 追踪
# ---------------------------------------------------------------------------

class TestEnvNotTracked:
    """.env 不在 git 仓库中"""

    def test_env_not_tracked_by_git(self):
        tracked = _git_ls_files()
        assert ".env" not in tracked, (
            ".env 被 git 追踪！请确认 .gitignore 包含 .env 规则"
        )


# ---------------------------------------------------------------------------
# 2. .env.example 被追踪
# ---------------------------------------------------------------------------

class TestEnvExampleTracked:
    """.env.example 作为模板被 git 追踪"""

    def test_env_example_is_tracked(self):
        tracked = _git_ls_files()
        assert ".env.example" in tracked, (
            ".env.example 未被 git 追踪！"
        )


# ---------------------------------------------------------------------------
# 3. .dockerignore 排除 .env
# ---------------------------------------------------------------------------

class TestDockerignoreExcludesEnv:
    """.dockerignore 排除 .env，防止打入镜像"""

    def test_dockerignore_exists(self):
        assert os.path.isfile(DOCKERIGNORE), (
            ".dockerignore 不存在！Dockerfile COPY . . 会把 .env 打入镜像"
        )

    def test_dockerignore_excludes_env(self):
        with open(DOCKERIGNORE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        excluded = False
        for pattern in lines:
            if fnmatch.fnmatch(".env", pattern):
                excluded = True
                break

        assert excluded, (
            f".dockerignore 未排除 .env！"
            f" 当前规则: {lines}"
        )
