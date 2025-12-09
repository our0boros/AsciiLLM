#!/usr/bin/env python3
# peek_db.py
"""
数据库状态监控工具 - 实时查看生成进度
支持在数据生成过程中安全地查看数据库状态
"""

import argparse
import json
import sqlite3
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import signal

# 可选依赖
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ==================== 数据库只读访问 ====================

class DBPeeker:
    """数据库只读查看器"""
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在: {self.db_path}")
    
    def _connect(self):
        """只读连接"""
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",  # 只读模式
            uri=True,
            timeout=5,
            isolation_level=None,  # 自动提交，避免锁
        )
        conn.row_factory = sqlite3.Row
        return conn
    
    def get_db_size(self) -> int:
        """获取数据库文件大小（字节）"""
        return self.db_path.stat().st_size
    
    def get_progress(self) -> dict:
        """获取生成进度"""
        conn = self._connect()
        try:
            cursor = conn.execute("SELECT key, value FROM progress")
            return {row['key']: row['value'] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            return {}
        finally:
            conn.close()
    
    def get_total_count(self) -> int:
        """获取总记录数"""
        conn = self._connect()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM samples")
            return cursor.fetchone()[0]
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()
    
    def get_stats_by_mode(self) -> list[dict]:
        """按模式统计"""
        conn = self._connect()
        try:
            cursor = conn.execute("""
                SELECT mode, COUNT(*) as count, 
                       AVG(w_char) as avg_width, 
                       AVG(h_char) as avg_height
                FROM samples 
                GROUP BY mode
            """)
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()
    
    def get_stats_by_long_edge(self) -> list[dict]:
        """按长边统计"""
        conn = self._connect()
        try:
            cursor = conn.execute("""
                SELECT long_edge, COUNT(*) as count
                FROM samples 
                GROUP BY long_edge
                ORDER BY long_edge
            """)
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()
    
    def get_recent_samples(self, n: int = 10) -> list[dict]:
        """获取最近的样本"""
        conn = self._connect()
        try:
            cursor = conn.execute(f"""
                SELECT id, prompt, orig_img, w_char, h_char, mode, long_edge, created_at
                FROM samples 
                ORDER BY id DESC 
                LIMIT {n}
            """)
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()
    
    def get_unique_images(self) -> int:
        """获取唯一图像数"""
        conn = self._connect()
        try:
            cursor = conn.execute("SELECT COUNT(DISTINCT orig_img) FROM samples")
            return cursor.fetchone()[0]
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()
    
    def get_sample_by_id(self, sample_id: int) -> Optional[dict]:
        """获取指定 ID 的样本"""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT * FROM samples WHERE id = ?", (sample_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.OperationalError:
            return None
        finally:
            conn.close()
    
    def get_random_sample(self) -> Optional[dict]:
        """获取随机样本"""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT * FROM samples ORDER BY RANDOM() LIMIT 1"
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.OperationalError:
            return None
        finally:
            conn.close()
    
    def get_generation_rate(self, window_minutes: int = 5) -> float:
        """计算最近的生成速率（条/分钟）"""
        conn = self._connect()
        try:
            cursor = conn.execute(f"""
                SELECT COUNT(*) as count
                FROM samples 
                WHERE created_at >= datetime('now', '-{window_minutes} minutes')
            """)
            count = cursor.fetchone()[0]
            return count / window_minutes if window_minutes > 0 else 0
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()
    
    def get_time_range(self) -> tuple[Optional[str], Optional[str]]:
        """获取数据时间范围"""
        conn = self._connect()
        try:
            cursor = conn.execute("""
                SELECT MIN(created_at) as first, MAX(created_at) as last
                FROM samples
            """)
            row = cursor.fetchone()
            return row['first'], row['last']
        except sqlite3.OperationalError:
            return None, None
        finally:
            conn.close()
    
    def execute_query(self, query: str, limit: int = 100) -> list[dict]:
        """执行自定义查询（只读）"""
        # 安全检查
        query_lower = query.lower().strip()
        if not query_lower.startswith('select'):
            raise ValueError("只允许 SELECT 查询")
        
        forbidden = ['insert', 'update', 'delete', 'drop', 'alter', 'create']
        for word in forbidden:
            if word in query_lower:
                raise ValueError(f"禁止使用 {word.upper()}")
        
        # 添加 LIMIT
        if 'limit' not in query_lower:
            query = f"{query} LIMIT {limit}"
        
        conn = self._connect()
        try:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()


# ==================== 输出格式化 ====================

def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """格式化时间间隔"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


# ==================== Rich 终端界面 ====================

class RichDisplay:
    """Rich 终端显示"""
    
    def __init__(self, peeker: DBPeeker):
        self.peeker = peeker
        self.console = Console()
    
    def show_stats(self):
        """显示统计信息"""
        # 基本信息
        total = self.peeker.get_total_count()
        unique_imgs = self.peeker.get_unique_images()
        db_size = self.peeker.get_db_size()
        progress = self.peeker.get_progress()
        rate = self.peeker.get_generation_rate(5)
        first, last = self.peeker.get_time_range()
        
        # 创建主表格
        table = Table(title="📊 数据库状态", box=box.ROUNDED)
        table.add_column("指标", style="cyan", width=20)
        table.add_column("值", style="green", width=30)
        
        table.add_row("总样本数", f"{total:,}")
        table.add_row("唯一图像数", f"{unique_imgs:,}")
        table.add_row("数据库大小", format_size(db_size))
        table.add_row("生成速率", f"{rate:.1f} 条/分钟")
        
        if progress:
            table.add_row("当前 Prompt 索引", str(progress.get('prompt_idx', 'N/A')))
            table.add_row("当前图像索引", str(progress.get('image_idx', 'N/A')))
        
        if first and last:
            table.add_row("首条记录时间", first)
            table.add_row("最新记录时间", last)
        
        self.console.print(table)
        
        # 按模式统计
        mode_stats = self.peeker.get_stats_by_mode()
        if mode_stats:
            mode_table = Table(title="📁 按模式统计", box=box.ROUNDED)
            mode_table.add_column("模式", style="cyan")
            mode_table.add_column("数量", style="green", justify="right")
            mode_table.add_column("平均宽度", style="yellow", justify="right")
            mode_table.add_column("平均高度", style="yellow", justify="right")
            
            for stat in mode_stats:
                mode_table.add_row(
                    stat['mode'],
                    f"{stat['count']:,}",
                    f"{stat['avg_width']:.1f}",
                    f"{stat['avg_height']:.1f}",
                )
            
            self.console.print(mode_table)
        
        # 按长边统计
        edge_stats = self.peeker.get_stats_by_long_edge()
        if edge_stats:
            edge_table = Table(title="📐 按长边统计", box=box.ROUNDED)
            edge_table.add_column("长边", style="cyan", justify="right")
            edge_table.add_column("数量", style="green", justify="right")
            
            for stat in edge_stats:
                edge_table.add_row(
                    str(stat['long_edge']),
                    f"{stat['count']:,}",
                )
            
            self.console.print(edge_table)
    
    def show_recent(self, n: int = 10):
        """显示最近的样本"""
        samples = self.peeker.get_recent_samples(n)
        
        if not samples:
            self.console.print("[yellow]暂无数据[/yellow]")
            return
        
        table = Table(title=f"🕐 最近 {n} 条记录", box=box.ROUNDED)
        table.add_column("ID", style="dim", width=8)
        table.add_column("图像", style="cyan", width=20)
        table.add_column("模式", style="green", width=10)
        table.add_column("长边", style="yellow", width=8)
        table.add_column("尺寸", style="magenta", width=12)
        table.add_column("Prompt", style="white", width=40, no_wrap=True)
        
        for s in samples:
            table.add_row(
                str(s['id']),
                s['orig_img'],
                s['mode'],
                str(s['long_edge']),
                f"{s['w_char']}×{s['h_char']}",
                s['prompt'][:40] + "..." if len(s['prompt']) > 40 else s['prompt'],
            )
        
        self.console.print(table)
    
    def show_sample(self, sample: dict):
        """显示单个样本详情"""
        self.console.print(Panel(
            f"""[cyan]ID:[/cyan] {sample['id']}
[cyan]Prompt:[/cyan] {sample['prompt']}
[cyan]Image:[/cyan] {sample['orig_img']}
[cyan]Mode:[/cyan] {sample['mode']}
[cyan]Long Edge:[/cyan] {sample['long_edge']}
[cyan]Size:[/cyan] {sample['w_char']}×{sample['h_char']} chars
[cyan]Created:[/cyan] {sample.get('created_at', 'N/A')}

[yellow]ASCII Preview:[/yellow]
{sample['ascii_text'][:500]}{'...' if len(sample['ascii_text']) > 500 else ''}
""",
            title="📄 样本详情",
            box=box.ROUNDED
        ))
    
    def watch(self, interval: float = 2.0):
        """实时监控模式"""
        self.console.print("[bold green]开始实时监控 (Ctrl+C 退出)[/bold green]")
        
        last_count = 0
        start_time = time.time()
        
        try:
            with Live(console=self.console, refresh_per_second=1) as live:
                while True:
                    # 获取数据
                    total = self.peeker.get_total_count()
                    unique_imgs = self.peeker.get_unique_images()
                    db_size = self.peeker.get_db_size()
                    progress = self.peeker.get_progress()
                    rate = self.peeker.get_generation_rate(1)
                    
                    # 计算增量
                    delta = total - last_count
                    last_count = total
                    
                    elapsed = time.time() - start_time
                    avg_rate = total / elapsed * 60 if elapsed > 0 else 0
                    
                    # 创建布局
                    layout = Layout()
                    
                    # 状态面板
                    status_text = Text()
                    status_text.append(f"📊 总样本数: ", style="cyan")
                    status_text.append(f"{total:,}", style="bold green")
                    status_text.append(f" (+{delta})\n", style="yellow")
                    
                    status_text.append(f"🖼️  唯一图像: ", style="cyan")
                    status_text.append(f"{unique_imgs:,}\n", style="green")
                    
                    status_text.append(f"💾 数据库大小: ", style="cyan")
                    status_text.append(f"{format_size(db_size)}\n", style="green")
                    
                    status_text.append(f"⚡ 实时速率: ", style="cyan")
                    status_text.append(f"{rate:.1f} 条/分钟\n", style="green")
                    
                    status_text.append(f"📈 平均速率: ", style="cyan")
                    status_text.append(f"{avg_rate:.1f} 条/分钟\n", style="green")
                    
                    status_text.append(f"⏱️  运行时间: ", style="cyan")
                    status_text.append(f"{format_duration(elapsed)}\n", style="green")
                    
                    if progress:
                        status_text.append(f"\n📍 进度: ", style="cyan")
                        status_text.append(
                            f"Prompt #{progress.get('prompt_idx', 0)}, "
                            f"Image #{progress.get('image_idx', 0)}",
                            style="yellow"
                        )
                    
                    # 最近样本
                    recent = self.peeker.get_recent_samples(3)
                    recent_text = Text()
                    for s in recent:
                        recent_text.append(f"#{s['id']} ", style="dim")
                        recent_text.append(f"[{s['mode']}/{s['long_edge']}] ", style="green")
                        recent_text.append(f"{s['prompt'][:50]}...\n", style="white")
                    
                    # 组合面板
                    panel = Panel(
                        status_text,
                        title=f"🔍 数据库监控 - {datetime.now().strftime('%H:%M:%S')}",
                        subtitle="Ctrl+C 退出",
                        box=box.DOUBLE
                    )
                    
                    live.update(panel)
                    time.sleep(interval)
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]监控已停止[/yellow]")


# ==================== 简单终端界面（无 Rich） ====================

class SimpleDisplay:
    """简单终端显示（无依赖）"""
    
    def __init__(self, peeker: DBPeeker):
        self.peeker = peeker
    
    def show_stats(self):
        """显示统计信息"""
        total = self.peeker.get_total_count()
        unique_imgs = self.peeker.get_unique_images()
        db_size = self.peeker.get_db_size()
        progress = self.peeker.get_progress()
        rate = self.peeker.get_generation_rate(5)
        first, last = self.peeker.get_time_range()
        
        print("\n" + "=" * 50)
        print("📊 数据库状态")
        print("=" * 50)
        print(f"总样本数:      {total:,}")
        print(f"唯一图像数:    {unique_imgs:,}")
        print(f"数据库大小:    {format_size(db_size)}")
        print(f"生成速率:      {rate:.1f} 条/分钟")
        
        if progress:
            print(f"当前进度:      Prompt #{progress.get('prompt_idx', 0)}, Image #{progress.get('image_idx', 0)}")
        
        if first and last:
            print(f"时间范围:      {first} ~ {last}")
        
        # 按模式统计
        mode_stats = self.peeker.get_stats_by_mode()
        if mode_stats:
            print("\n📁 按模式统计:")
            print("-" * 40)
            print(f"{'模式':<15} {'数量':>10} {'平均宽度':>10} {'平均高度':>10}")
            print("-" * 40)
            for stat in mode_stats:
                print(f"{stat['mode']:<15} {stat['count']:>10,} {stat['avg_width']:>10.1f} {stat['avg_height']:>10.1f}")
        
        # 按长边统计
        edge_stats = self.peeker.get_stats_by_long_edge()
        if edge_stats:
            print("\n📐 按长边统计:")
            print("-" * 25)
            print(f"{'长边':>10} {'数量':>10}")
            print("-" * 25)
            for stat in edge_stats:
                print(f"{stat['long_edge']:>10} {stat['count']:>10,}")
        
        print("=" * 50 + "\n")
    
    def show_recent(self, n: int = 10):
        """显示最近的样本"""
        samples = self.peeker.get_recent_samples(n)
        
        if not samples:
            print("暂无数据")
            return
        
        print(f"\n🕐 最近 {n} 条记录:")
        print("-" * 80)
        print(f"{'ID':>8} {'图像':<20} {'模式':<10} {'长边':>6} {'尺寸':<12} Prompt")
        print("-" * 80)
        
        for s in samples:
            prompt = s['prompt'][:35] + "..." if len(s['prompt']) > 35 else s['prompt']
            print(f"{s['id']:>8} {s['orig_img']:<20} {s['mode']:<10} {s['long_edge']:>6} {s['w_char']}×{s['h_char']:<8} {prompt}")
        
        print("-" * 80 + "\n")
    
    def show_sample(self, sample: dict):
        """显示单个样本详情"""
        print("\n" + "=" * 60)
        print("📄 样本详情")
        print("=" * 60)
        print(f"ID:        {sample['id']}")
        print(f"Prompt:    {sample['prompt']}")
        print(f"Image:     {sample['orig_img']}")
        print(f"Mode:      {sample['mode']}")
        print(f"Long Edge: {sample['long_edge']}")
        print(f"Size:      {sample['w_char']}×{sample['h_char']} chars")
        print(f"Created:   {sample.get('created_at', 'N/A')}")
        print("\nASCII Preview:")
        print("-" * 60)
        print(sample['ascii_text'][:800])
        if len(sample['ascii_text']) > 800:
            print("... (截断)")
        print("=" * 60 + "\n")
    
    def watch(self, interval: float = 2.0):
        """实时监控模式"""
        print("开始实时监控 (Ctrl+C 退出)\n")
        
        last_count = 0
        start_time = time.time()
        
        try:
            while True:
                # 清屏
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # 获取数据
                total = self.peeker.get_total_count()
                unique_imgs = self.peeker.get_unique_images()
                db_size = self.peeker.get_db_size()
                progress = self.peeker.get_progress()
                rate = self.peeker.get_generation_rate(1)
                
                delta = total - last_count
                last_count = total
                
                elapsed = time.time() - start_time
                avg_rate = total / elapsed * 60 if elapsed > 0 else 0
                
                # 显示
                now = datetime.now().strftime('%H:%M:%S')
                print(f"🔍 数据库监控 - {now}")
                print("=" * 40)
                print(f"📊 总样本数:    {total:,} (+{delta})")
                print(f"🖼️  唯一图像:    {unique_imgs:,}")
                print(f"💾 数据库大小:  {format_size(db_size)}")
                print(f"⚡ 实时速率:    {rate:.1f} 条/分钟")
                print(f"📈 平均速率:    {avg_rate:.1f} 条/分钟")
                print(f"⏱️  运行时间:    {format_duration(elapsed)}")
                
                if progress:
                    print(f"📍 进度:        Prompt #{progress.get('prompt_idx', 0)}, Image #{progress.get('image_idx', 0)}")
                
                print("\nCtrl+C 退出")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='数据库状态监控工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s stats data/dataset.db          # 显示统计信息
  %(prog)s recent data/dataset.db -n 20   # 显示最近20条
  %(prog)s watch data/dataset.db          # 实时监控
  %(prog)s sample data/dataset.db         # 随机抽样
  %(prog)s query data/dataset.db "SELECT mode, COUNT(*) FROM samples GROUP BY mode"
        """
    )
    
    parser.add_argument(
        'command',
        choices=['stats', 'recent', 'watch', 'sample', 'query', 'get'],
        help='命令: stats(统计), recent(最近), watch(监控), sample(抽样), query(查询), get(指定ID)'
    )
    parser.add_argument(
        '-d', '--db_path',
        type=str,
        default='data/ascii_art_dataset/dataset.db',
        help='SQLite 数据库路径'
    )
    parser.add_argument(
        'query_str',
        nargs='?',
        default=None,
        help='SQL 查询语句 (仅用于 query 命令)'
    )
    parser.add_argument(
        '-n', '--number',
        type=int,
        default=10,
        help='显示数量 (默认: 10)'
    )
    parser.add_argument(
        '-i', '--interval',
        type=float,
        default=2.0,
        help='监控刷新间隔秒数 (默认: 2.0)'
    )
    parser.add_argument(
        '--id',
        type=int,
        default=None,
        help='样本 ID (用于 get 命令)'
    )
    parser.add_argument(
        '--no-rich',
        action='store_true',
        help='禁用 Rich 终端界面'
    )
    
    args = parser.parse_args()
    
    # 创建查看器
    try:
        peeker = DBPeeker(Path(args.db_path))
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 选择显示方式
    use_rich = HAS_RICH and not args.no_rich
    display = RichDisplay(peeker) if use_rich else SimpleDisplay(peeker)
    
    # 执行命令
    if args.command == 'stats':
        display.show_stats()
    
    elif args.command == 'recent':
        display.show_recent(args.number)
    
    elif args.command == 'watch':
        display.watch(args.interval)
    
    elif args.command == 'sample':
        for _ in range(args.number):
            sample = peeker.get_random_sample()
            if sample:
                display.show_sample(sample)
            else:
                print("暂无数据")
                break
    
    elif args.command == 'get':
        if args.id is None:
            print("错误: 请使用 --id 指定样本 ID")
            sys.exit(1)
        sample = peeker.get_sample_by_id(args.id)
        if sample:
            display.show_sample(sample)
        else:
            print(f"未找到 ID={args.id} 的样本")
    
    elif args.command == 'query':
        if not args.query_str:
            print("错误: 请提供 SQL 查询语句")
            sys.exit(1)
        try:
            results = peeker.execute_query(args.query_str, args.number)
            if results:
                # 简单表格输出
                headers = list(results[0].keys())
                print("\t".join(headers))
                print("-" * 60)
                for row in results:
                    print("\t".join(str(v) for v in row.values()))
            else:
                print("无结果")
        except ValueError as e:
            print(f"错误: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()

# # 安装可选依赖（推荐，更好的界面）
# pip install rich

# # 1. 查看统计信息
# python peek_db.py stats data/ascii_art_dataset/dataset.db

# # 2. 查看最近 20 条记录
# python peek_db.py recent data/ascii_art_dataset/dataset.db -n 20

# # 3. 实时监控（每 1 秒刷新）
# python peek_db.py watch data/ascii_art_dataset/dataset.db -i 1

# # 4. 随机抽样 3 条记录
# python peek_db.py sample data/ascii_art_dataset/dataset.db -n 3

# # 5. 查看指定 ID 的样本
# python peek_db.py get data/ascii_art_dataset/dataset.db --id 12345

# # 6. 自定义 SQL 查询
# python peek_db.py query data/ascii_art_dataset/dataset.db \
#     "SELECT mode, long_edge, COUNT(*) as cnt FROM samples GROUP BY mode, long_edge"

# # 7. 禁用 Rich 界面（纯文本）
# python peek_db.py stats data/ascii_art_dataset/dataset.db --no-rich
