#!/usr/bin/env python3
# export_to_parquet.py
"""
将 SQLite 数据库导出为分包 Parquet 文件
特性：流式处理、内存安全、自动分包
"""

import argparse
import json
import sqlite3
import logging
from pathlib import Path
from typing import Iterator, Optional
import gc

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParquetExporter:
    """SQLite 到分包 Parquet 导出器"""
    
    def __init__(
        self,
        db_path: Path,
        output_dir: Path,
        max_size_gb: float = 10.0,
        chunk_size: int = 10000,
        compression: str = 'snappy',
    ):
        """
        Args:
            db_path: SQLite 数据库路径
            output_dir: 输出目录
            max_size_gb: 每个 parquet 文件的最大大小（GB）
            chunk_size: 每次从数据库读取的行数
            compression: 压缩算法 (snappy/gzip/zstd/none)
        """
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.chunk_size = chunk_size
        self.compression = compression if compression != 'none' else None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在: {self.db_path}")
    
    def _get_total_rows(self) -> int:
        """获取总行数"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM samples")
            return cursor.fetchone()[0]
    
    def _iter_chunks(self) -> Iterator[pd.DataFrame]:
        """流式读取数据库"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM samples"
        
        for chunk in pd.read_sql_query(
            query, conn, chunksize=self.chunk_size
        ):
            # 反序列化 colors 字段
            if 'colors' in chunk.columns:
                chunk['colors'] = chunk['colors'].apply(
                    lambda x: json.loads(x) if pd.notna(x) and x else None
                )
            yield chunk
        
        conn.close()
    
    def _estimate_size(self, df: pd.DataFrame) -> int:
        """估算 DataFrame 的 Parquet 大小（字节）"""
        # 转换为 Arrow Table 并估算大小
        table = pa.Table.from_pandas(df)
        # 使用内存大小作为估算（实际压缩后会更小）
        return table.nbytes
    
    def _write_parquet(self, df: pd.DataFrame, part_idx: int) -> Path:
        """写入单个 Parquet 文件"""
        output_path = self.output_dir / f"part_{part_idx:04d}.parquet"
        
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression=self.compression,
            index=False,
        )
        
        return output_path
    
    def export(self) -> list[Path]:
        """执行导出"""
        total_rows = self._get_total_rows()
        logger.info(f"总记录数: {total_rows:,}")
        logger.info(f"分包大小限制: {self.max_size_bytes / 1024**3:.1f} GB")
        logger.info(f"压缩算法: {self.compression or '无'}")
        
        output_files = []
        current_dfs = []
        current_size = 0
        part_idx = 0
        
        pbar = tqdm(total=total_rows, desc="导出进度", unit="行")
        
        for chunk in self._iter_chunks():
            chunk_size = self._estimate_size(chunk)
            
            # 检查是否需要写入当前分包
            if current_size + chunk_size > self.max_size_bytes and current_dfs:
                # 合并并写入
                combined_df = pd.concat(current_dfs, ignore_index=True)
                output_path = self._write_parquet(combined_df, part_idx)
                
                actual_size = output_path.stat().st_size
                logger.info(
                    f"写入分包 {part_idx}: {len(combined_df):,} 行, "
                    f"{actual_size / 1024**3:.2f} GB -> {output_path.name}"
                )
                
                output_files.append(output_path)
                part_idx += 1
                
                # 清理内存
                del current_dfs, combined_df
                gc.collect()
                
                current_dfs = []
                current_size = 0
            
            current_dfs.append(chunk)
            current_size += chunk_size
            pbar.update(len(chunk))
        
        # 写入最后一个分包
        if current_dfs:
            combined_df = pd.concat(current_dfs, ignore_index=True)
            output_path = self._write_parquet(combined_df, part_idx)
            
            actual_size = output_path.stat().st_size
            logger.info(
                f"写入分包 {part_idx}: {len(combined_df):,} 行, "
                f"{actual_size / 1024**3:.2f} GB -> {output_path.name}"
            )
            
            output_files.append(output_path)
        
        pbar.close()
        
        # 输出汇总信息
        self._print_summary(output_files)
        
        return output_files
    
    def _print_summary(self, files: list[Path]):
        """打印汇总信息"""
        total_size = sum(f.stat().st_size for f in files)
        
        logger.info("=" * 50)
        logger.info("导出完成！")
        logger.info(f"分包数量: {len(files)}")
        logger.info(f"总大小: {total_size / 1024**3:.2f} GB")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("文件列表:")
        for f in files:
            size_gb = f.stat().st_size / 1024**3
            logger.info(f"  - {f.name}: {size_gb:.2f} GB")


class ParquetValidator:
    """验证导出的 Parquet 文件"""
    
    @staticmethod
    def validate(parquet_dir: Path) -> bool:
        """验证所有分包文件"""
        files = sorted(parquet_dir.glob("part_*.parquet"))
        
        if not files:
            logger.error("未找到 parquet 文件")
            return False
        
        logger.info(f"验证 {len(files)} 个分包文件...")
        
        total_rows = 0
        errors = []
        
        for f in tqdm(files, desc="验证中"):
            try:
                # 读取元数据（不加载全部数据）
                parquet_file = pq.ParquetFile(f)
                num_rows = parquet_file.metadata.num_rows
                total_rows += num_rows
                
                # 抽样验证（读取前100行）
                df_sample = parquet_file.read_row_group(0).to_pandas().head(100)
                
                # 检查必要列
                required_cols = ['prompt', 'orig_img', 'ascii_text', 'mode']
                missing = set(required_cols) - set(df_sample.columns)
                if missing:
                    errors.append(f"{f.name}: 缺少列 {missing}")
                    
            except Exception as e:
                errors.append(f"{f.name}: {e}")
        
        if errors:
            logger.error("验证失败:")
            for err in errors:
                logger.error(f"  - {err}")
            return False
        
        logger.info(f"验证通过！总行数: {total_rows:,}")
        return True
    
    @staticmethod
    def sample(parquet_dir: Path, n: int = 5):
        """随机抽样查看数据"""
        files = sorted(parquet_dir.glob("part_*.parquet"))
        
        if not files:
            logger.error("未找到 parquet 文件")
            return
        
        # 从第一个文件抽样
        df = pd.read_parquet(files[0])
        samples = df.sample(min(n, len(df)))
        
        logger.info(f"随机抽样 {len(samples)} 条记录:")
        for idx, row in samples.iterrows():
            logger.info("-" * 40)
            logger.info(f"Prompt: {row['prompt'][:80]}...")
            logger.info(f"Image: {row['orig_img']}")
            logger.info(f"Mode: {row['mode']}, Long Edge: {row['long_edge']}")
            logger.info(f"Size: {row['w_char']}x{row['h_char']} chars")
            logger.info(f"ASCII Preview:\n{row['ascii_text'][:200]}...")


def merge_parquets(parquet_dir: Path, output_path: Path):
    """将分包合并为单个文件（如果需要）"""
    files = sorted(parquet_dir.glob("part_*.parquet"))
    
    if not files:
        logger.error("未找到 parquet 文件")
        return
    
    logger.info(f"合并 {len(files)} 个分包...")
    
    # 使用 pyarrow 流式合并
    with pq.ParquetWriter(output_path, pq.read_schema(files[0])) as writer:
        for f in tqdm(files, desc="合并中"):
            table = pq.read_table(f)
            writer.write_table(table)
    
    logger.info(f"合并完成: {output_path}")
    logger.info(f"文件大小: {output_path.stat().st_size / 1024**3:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description='将 SQLite 数据库导出为分包 Parquet 文件'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 导出命令
    export_parser = subparsers.add_parser('export', help='导出数据库')
    export_parser.add_argument(
        'db_path',
        type=str,
        help='SQLite 数据库路径'
    )
    export_parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./parquet_output',
        help='输出目录 (默认: ./parquet_output)'
    )
    export_parser.add_argument(
        '-s', '--max-size',
        type=float,
        default=10.0,
        help='每个分包的最大大小（GB）(默认: 10.0)'
    )
    export_parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=10000,
        help='每次读取的行数 (默认: 10000)'
    )
    export_parser.add_argument(
        '--compression',
        type=str,
        choices=['snappy', 'gzip', 'zstd', 'none'],
        default='snappy',
        help='压缩算法 (默认: snappy)'
    )
    
    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='验证 parquet 文件')
    validate_parser.add_argument(
        'parquet_dir',
        type=str,
        help='Parquet 文件目录'
    )
    validate_parser.add_argument(
        '--sample',
        type=int,
        default=0,
        help='随机抽样显示的记录数 (默认: 0，不抽样)'
    )
    
    # 合并命令
    merge_parser = subparsers.add_parser('merge', help='合并分包文件')
    merge_parser.add_argument(
        'parquet_dir',
        type=str,
        help='Parquet 分包目录'
    )
    merge_parser.add_argument(
        '-o', '--output',
        type=str,
        default='./merged.parquet',
        help='输出文件路径 (默认: ./merged.parquet)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'export':
        exporter = ParquetExporter(
            db_path=Path(args.db_path),
            output_dir=Path(args.output_dir),
            max_size_gb=args.max_size,
            chunk_size=args.chunk_size,
            compression=args.compression,
        )
        exporter.export()
        
    elif args.command == 'validate':
        parquet_dir = Path(args.parquet_dir)
        ParquetValidator.validate(parquet_dir)
        if args.sample > 0:
            ParquetValidator.sample(parquet_dir, args.sample)
            
    elif args.command == 'merge':
        merge_parquets(
            parquet_dir=Path(args.parquet_dir),
            output_path=Path(args.output),
        )
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()


# # 1. 导出（自动分包，每包 10GB 以内）
# python export_to_parquet.py export data/ascii_art_dataset/dataset.db \
#     -o data/parquet_output \
#     -s 10.0 \
#     --compression snappy

# # 2. 验证导出结果
# python export_to_parquet.py validate data/parquet_output --sample 5

# # 3. 如需合并（可选）
# python export_to_parquet.py merge data/parquet_output -o data/full_dataset.parquet
