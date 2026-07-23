"""
数据库初始化脚本
===============

用于创建MySQL数据库和表结构。

使用方法：
    python init_db.py

注意：
    运行此脚本前，请确保：
    1. MySQL服务已启动
    2. 已在.env中配置正确的数据库连接信息
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from config.settings import settings
from schemas.database import Base, db_manager


def create_database():
    """
    创建数据库（如果不存在）
    
    使用root连接MySQL服务器，创建应用数据库。
    """
    from sqlalchemy import create_engine
    
    # 连接到MySQL服务器（不指定数据库）
    server_url = (
        f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
        f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}"
    )
    
    try:
        engine = create_engine(server_url, echo=True)
        with engine.connect() as conn:
            # 创建数据库（如果不存在）
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {settings.MYSQL_DATABASE} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
            conn.commit()
            print(f"✓ 数据库 '{settings.MYSQL_DATABASE}' 创建成功")
    except Exception as e:
        print(f"✗ 创建数据库失败: {e}")
        return False
    
    return True


def create_tables():
    """
    创建数据表
    
    使用SQLAlchemy的ORM模型创建表结构。
    """
    try:
        db_manager.create_tables()
        print("✓ 数据表创建成功")
        return True
    except Exception as e:
        print(f"✗ 创建数据表失败: {e}")
        return False


def show_tables():
    """显示创建的表"""
    from sqlalchemy import inspect
    
    inspector = inspect(db_manager.engine)
    tables = inspector.get_table_names()
    
    print("\n已创建的表：")
    for table in tables:
        print(f"  - {table}")


def main():
    """主函数"""
    print("=" * 50)
    print("心理咨询AI - 数据库初始化")
    print("=" * 50)
    
    print(f"\n数据库配置：")
    print(f"  主机: {settings.MYSQL_HOST}")
    print(f"  端口: {settings.MYSQL_PORT}")
    print(f"  用户: {settings.MYSQL_USER}")
    print(f"  数据库: {settings.MYSQL_DATABASE}")
    
    print("\n开始初始化...\n")
    
    # 步骤1：创建数据库
    if not create_database():
        sys.exit(1)
    
    # 步骤2：创建表
    if not create_tables():
        sys.exit(1)
    
    # 步骤3：显示结果
    show_tables()
    
    print("\n" + "=" * 50)
    print("数据库初始化完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
