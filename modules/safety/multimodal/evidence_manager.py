"""
证据管理模块
功能：管理违规证据的保存、查询、清理
"""

import os
import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# 项目根目录（用于将相对路径转换为绝对路径，兼容 API 服务和直接运行）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class EvidenceManager:
    """证据管理器"""

    def __init__(self, evidence_dir: str = "evidence", max_storage_mb: int = 500):
        """
        初始化证据管理器

        Args:
            evidence_dir: 证据存储目录
            max_storage_mb: 最大存储空间（MB）
        """
        # 将相对路径转为基于项目根的绝对路径，避免 CWD 不同导致路径错乱
        if evidence_dir and not os.path.isabs(evidence_dir):
            evidence_dir = os.path.join(_PROJECT_ROOT, evidence_dir)
        self.evidence_dir = Path(evidence_dir)
        self.max_storage_mb = max_storage_mb

        # 创建证据目录
        self.evidence_dir.mkdir(exist_ok=True)

        # 子目录
        self.frames_dir = self.evidence_dir / "frames"
        self.audio_dir = self.evidence_dir / "audio"
        self.meta_dir = self.evidence_dir / "meta"

        for d in [self.frames_dir, self.audio_dir, self.meta_dir]:
            d.mkdir(exist_ok=True)

    def save_frame_evidence(
        self,
        frame,
        violation_type: str,
        timestamp: float = None,
        metadata: Dict = None
    ) -> Dict:
        """
        保存违规帧证据

        Args:
            frame: 图像帧（numpy数组或PIL Image）
            violation_type: 违规类型
            timestamp: 时间戳
            metadata: 额外元数据

        Returns:
            保存结果字典
        """
        import cv2

        # 生成文件名
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if timestamp is not None:
            ts = f"{ts}_t{timestamp:.2f}s"

        filename = f"{ts}_{violation_type}.jpg"
        filepath = self.frames_dir / filename

        try:
            # 保存图像
            if isinstance(frame, Path) or isinstance(frame, str):
                # 如果是路径，复制文件
                shutil.copy(str(frame), filepath)
            else:
                # 如果是numpy数组，用 imencode + tofile 保存（避免 cv2.imwrite 在 Windows 下不支持中文路径）
                _, buf = cv2.imencode(os.path.splitext(str(filepath))[1], frame)
                buf.tofile(str(filepath))

            # 保存元数据
            meta = {
                "filename": filename,
                "violation_type": violation_type,
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "file_size": filepath.stat().st_size,
                **(metadata or {})
            }

            meta_path = self.meta_dir / f"{filename}.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info(f"保存帧证据: {filepath}")

            return {
                "success": True,
                "path": str(filepath),
                "meta_path": str(meta_path),
                "filename": filename
            }

        except Exception as e:
            logger.error(f"保存帧证据失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def save_audio_evidence(
        self,
        audio_path: str,
        violation_keywords: List[str],
        transcript: str = "",
        timestamp: float = None,
        metadata: Dict = None
    ) -> Dict:
        """
        保存违规音频证据

        Args:
            audio_path: 音频文件路径
            violation_keywords: 违规关键词列表
            transcript: 转录文本
            timestamp: 时间戳
            metadata: 额外元数据

        Returns:
            保存结果字典
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if timestamp is not None:
            ts = f"{ts}_t{timestamp:.2f}s"

        filename = f"{ts}_audio.wav"
        filepath = self.audio_dir / filename

        try:
            # 复制音频文件
            shutil.copy(audio_path, filepath)

            # 保存元数据
            meta = {
                "filename": filename,
                "violation_keywords": violation_keywords,
                "transcript": transcript,
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat(),
                "file_size": filepath.stat().st_size,
                **(metadata or {})
            }

            meta_path = self.meta_dir / f"{filename}.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info(f"保存音频证据: {filepath}")

            return {
                "success": True,
                "path": str(filepath),
                "meta_path": str(meta_path),
                "filename": filename
            }

        except Exception as e:
            logger.error(f"保存音频证据失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def save_full_report(
        self,
        result: Dict,
        video_path: str = None,
        user_id: str = None
    ) -> Dict:
        """
        保存完整的检测报告

        Args:
            result: 检测结果
            video_path: 视频路径
            user_id: 用户ID

        Returns:
            保存结果
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_report.json"
        filepath = self.meta_dir / filename

        report = {
            "report_id": ts,
            "created_at": datetime.now().isoformat(),
            "video_path": video_path,
            "user_id": user_id,
            "result": result
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"保存检测报告: {filepath}")

            return {
                "success": True,
                "path": str(filepath),
                "report_id": ts
            }

        except Exception as e:
            logger.error(f"保存报告失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def list_evidence(
        self,
        evidence_type: str = None,
        violation_type: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        查询证据列表

        Args:
            evidence_type: 证据类型 (frames/audio/meta)
            violation_type: 违规类型
            start_date: 开始日期
            end_date: 结束日期
            limit: 返回数量限制

        Returns:
            证据列表
        """
        evidence_list = []

        # 遍历元数据目录
        for meta_file in sorted(self.meta_dir.glob("*.json"), reverse=True):
            if len(evidence_list) >= limit:
                break

            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                # 过滤条件
                if violation_type and meta.get("violation_type") != violation_type:
                    continue

                if start_date or end_date:
                    created_at = datetime.fromisoformat(meta.get("created_at", ""))
                    if start_date and created_at < start_date:
                        continue
                    if end_date and created_at > end_date:
                        continue

                meta["meta_path"] = str(meta_file)
                evidence_list.append(meta)

            except Exception as e:
                logger.warning(f"读取元数据失败: {meta_file}, {e}")

        return evidence_list

    def get_evidence(self, report_id: str) -> Optional[Dict]:
        """
        获取特定证据详情

        Args:
            report_id: 报告ID

        Returns:
            证据详情
        """
        meta_file = self.meta_dir / f"{report_id}_report.json"
        if not meta_file.exists():
            # 尝试查找其他类型的元数据
            for f in self.meta_dir.glob(f"*{report_id}*.json"):
                meta_file = f
                break

        if not meta_file.exists():
            return None

        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"读取证据失败: {e}")
            return None

    def delete_evidence(self, report_id: str) -> bool:
        """
        删除证据

        Args:
            report_id: 报告ID

        Returns:
            是否成功
        """
        try:
            deleted = False

            # 删除帧证据
            for f in self.frames_dir.glob(f"*{report_id}*"):
                f.unlink()
                deleted = True

            # 删除音频证据
            for f in self.audio_dir.glob(f"*{report_id}*"):
                f.unlink()
                deleted = True

            # 删除元数据
            for f in self.meta_dir.glob(f"*{report_id}*"):
                f.unlink()
                deleted = True

            return deleted

        except Exception as e:
            logger.error(f"删除证据失败: {e}")
            return False

    def cleanup_old_evidence(self, days: int = 30) -> int:
        """
        清理过期证据

        Args:
            days: 保留天数

        Returns:
            删除的文件数量
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0

        # 清理帧证据
        for f in self.frames_dir.glob("*.jpg"):
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff_date:
                f.unlink()
                deleted_count += 1

        # 清理音频证据
        for f in self.audio_dir.glob("*.wav"):
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff_date:
                f.unlink()
                deleted_count += 1

        # 清理元数据
        for f in self.meta_dir.glob("*.json"):
            if datetime.fromtimestamp(f.stat().st_mtime) < cutoff_date:
                f.unlink()
                deleted_count += 1

        logger.info(f"清理了 {deleted_count} 个过期证据文件")
        return deleted_count

    def get_storage_usage(self) -> Dict:
        """
        获取存储使用情况

        Returns:
            存储统计信息
        """
        def get_dir_size(path: Path) -> int:
            total = 0
            for f in path.glob("**/*"):
                if f.is_file():
                    total += f.stat().st_size
            return total

        frames_size = get_dir_size(self.frames_dir)
        audio_size = get_dir_size(self.audio_dir)
        meta_size = get_dir_size(self.meta_dir)
        total_size = frames_size + audio_size + meta_size

        return {
            "frames_count": len(list(self.frames_dir.glob("*.jpg"))),
            "audio_count": len(list(self.audio_dir.glob("*.wav"))),
            "reports_count": len(list(self.meta_dir.glob("*_report.json"))),
            "frames_size_mb": round(frames_size / 1024 / 1024, 2),
            "audio_size_mb": round(audio_size / 1024 / 1024, 2),
            "meta_size_mb": round(meta_size / 1024 / 1024, 2),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "max_storage_mb": self.max_storage_mb,
            "usage_percent": round(total_size / 1024 / 1024 / self.max_storage_mb * 100, 2)
        }


# ============ 简单测试 ============
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("证据管理器测试")
    print("=" * 60)

    # 初始化
    manager = EvidenceManager(evidence_dir="evidence")

    # 查看存储使用情况
    usage = manager.get_storage_usage()
    print(f"\n存储使用情况:")
    print(f"  帧证据: {usage['frames_count']} 个, {usage['frames_size_mb']} MB")
    print(f"  音频证据: {usage['audio_count']} 个, {usage['audio_size_mb']} MB")
    print(f"  报告: {usage['reports_count']} 个")
    print(f"  总计: {usage['total_size_mb']} MB / {usage['max_storage_mb']} MB ({usage['usage_percent']}%)")

    # 查询证据列表
    print(f"\n最近的证据:")
    evidence_list = manager.list_evidence(limit=5)
    for e in evidence_list:
        print(f"  - {e.get('created_at', 'N/A')}: {e.get('violation_type', e.get('filename', 'N/A'))}")
