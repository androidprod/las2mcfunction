#!/usr/bin/env python3
"""
las2mcfunction_enhanced.py

LAS/ZIP → Minecraft setblock マクロ生成（改良版）
特徴:
 - 引数は argparse で安全に処理（パスに空白があってもOK）
 - スレッド同期に threading.Lock を使用（内部属性に依存しない）
 - きれいなプログレス表示（rich があればそちらを優先、無ければ tqdm、最後はテキスト）
 - 詳細なデバッグ/ログ出力（--debug で DEBUG レベル）
 - ファイル単位の堅牢なエラーハンドリング（1ファイルの失敗が全体を止めない）
 - GPU（OpenCL）使用オプション、Numba を使った高速 CPU パス（利用可能なら）

使い方例:
 python las2mcfunction_enhanced.py "D:/TenGun/DL_DATA" "C:/Users/droid/.../TenGun/functions" --threads 4 --gpu 1 --max-lines 10000 --scale 1 --debug

依存: laspy, numpy, numba (任意), pyopencl (任意), rich (任意), tqdm (任意)
インストール例:
 pip install laspy numpy numba pyopencl rich tqdm

"""

import argparse
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 数値処理
import numpy as np
try:
    import laspy
except Exception as e:
    print("laspy が見つかりません。pip install laspy を実行してください。", file=sys.stderr)
    raise

# OpenCL (任意)
try:
    import pyopencl as cl
    GPU_AVAILABLE = True
except Exception:
    cl = None
    GPU_AVAILABLE = False

# Numba（任意）
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# プログレス表示: rich -> tqdm -> テキスト
USE_RICH = False
USE_TQDM = False
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskID
    from rich.console import Console
    USE_RICH = True
    console = Console()
except Exception:
    try:
        from tqdm import tqdm
        USE_TQDM = True
    except Exception:
        pass

import threading
from collections import namedtuple
RGB = namedtuple("RGB", ["r", "g", "b"]) 

# ----------------------------
# カラーマップ（オリジナルを踏襲）
COLOR_MAP = [
    ("white_concrete", RGB(207,213,214)), ("orange_concrete", RGB(224,97,0)),
    ("magenta_concrete", RGB(169,48,159)), ("light_blue_concrete", RGB(36,137,199)),
    ("yellow_concrete", RGB(241,175,21)), ("lime_concrete", RGB(94,168,24)),
    ("pink_concrete", RGB(214,101,143)), ("gray_concrete", RGB(54,57,61)),
    ("light_gray_concrete", RGB(125,125,115)), ("cyan_concrete", RGB(21,119,136)),
    ("purple_concrete", RGB(100,32,156)), ("blue_concrete", RGB(45,47,143)),
    ("brown_concrete", RGB(96,60,32)), ("green_concrete", RGB(73,91,36)),
    ("red_concrete", RGB(142,33,33)), ("black_concrete", RGB(8,10,15)),

    ("white_wool", RGB(233,236,236)), ("orange_wool", RGB(240,118,19)),
    ("magenta_wool", RGB(190,68,201)), ("light_blue_wool", RGB(58,175,217)),
    ("yellow_wool", RGB(249,199,35)), ("lime_wool", RGB(112,185,25)),
    ("pink_wool", RGB(237,141,172)), ("gray_wool", RGB(62,68,71)),
    ("light_gray_wool", RGB(142,142,134)), ("cyan_wool", RGB(21,137,145)),
    ("purple_wool", RGB(122,42,172)), ("blue_wool", RGB(53,57,157)),
    ("brown_wool", RGB(114,71,40)), ("green_wool", RGB(85,110,27)),
    ("red_wool", RGB(162,34,35)), ("black_wool", RGB(21,21,26)),

    ("white_terracotta", RGB(209,178,161)), ("orange_terracotta", RGB(161,83,37)),
    ("magenta_terracotta", RGB(150,88,109)), ("light_blue_terracotta", RGB(113,108,137)),
    ("yellow_terracotta", RGB(186,133,35)), ("lime_terracotta", RGB(103,117,52)),
    ("pink_terracotta", RGB(160,77,78)), ("gray_terracotta", RGB(57,42,35)),
    ("light_gray_terracotta", RGB(135,107,98)), ("cyan_terracotta", RGB(87,91,91)),
    ("purple_terracotta", RGB(118,70,86)), ("blue_terracotta", RGB(74,59,91)),
    ("brown_terracotta", RGB(77,51,36)), ("green_terracotta", RGB(76,82,42)),
    ("red_terracotta", RGB(143,61,47)), ("black_terracotta", RGB(37,23,16))
]

COLOR_ARRAY = np.array([[c.r, c.g, c.b] for _, c in COLOR_MAP], dtype=np.int32)

# ----------------------------
# OpenCL カーネル（そのまま使用）
OPENCL_KERNEL = r"""
__kernel void closest_block(
    __global const int* R, __global const int* G, __global const int* B,
    __global const int* colors, __global int* out, const int n_blocks)
{
    int i = get_global_id(0);
    int min_idx = 0;
    int min_dist = 2147483647;
    int r = R[i], g = G[i], b = B[i];
    for(int j=0;j<n_blocks;j++){
        int dr = r - colors[j*3+0];
        int dg = g - colors[j*3+1];
        int db = b - colors[j*3+2];
        int d = dr*dr + dg*dg + db*db;
        if(d<min_dist){ min_dist=d; min_idx=j;}
    }
    out[i]=min_idx;
}
"""

# OpenCL 実行関数
def closest_block_opencl(R, G, B):
    if not GPU_AVAILABLE:
        raise RuntimeError("OpenCL が利用できません")
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    n_points = R.shape[0]
    n_blocks = COLOR_ARRAY.shape[0]

    colors_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=COLOR_ARRAY.flatten())
    R_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=R)
    G_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=G)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    out_buf = cl.Buffer(ctx, mf.WRITE_ONLY, R.nbytes)

    prg = cl.Program(ctx, OPENCL_KERNEL).build()
    prg.closest_block(queue, (n_points,), None, R_buf, G_buf, B_buf, colors_buf, out_buf, np.int32(n_blocks))
    out = np.empty_like(R)
    cl.enqueue_copy(queue, out, out_buf)
    return out

# ----------------------------
# CPU パス: numba 使用可
if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def closest_block_cpu(R, G, B):
        n_points = R.shape[0]
        n_blocks = COLOR_ARRAY.shape[0]
        result = np.empty(n_points, dtype=np.int32)
        for i in prange(n_points):
            min_dist = 1e12
            best = 0
            for j in range(n_blocks):
                dr = int(R[i]) - COLOR_ARRAY[j,0]
                dg = int(G[i]) - COLOR_ARRAY[j,1]
                db = int(B[i]) - COLOR_ARRAY[j,2]
                d = dr*dr + dg*dg + db*db
                if d<min_dist:
                    min_dist = d
                    best = j
            result[i] = best
        return result
else:
    def closest_block_cpu(R, G, B):
        # シンプルだが確実な numpy ベースの代替
        n_points = R.shape[0]
        result = np.empty(n_points, dtype=np.int32)
        for i in range(n_points):
            dr = COLOR_ARRAY[:,0] - int(R[i])
            dg = COLOR_ARRAY[:,1] - int(G[i])
            db = COLOR_ARRAY[:,2] - int(B[i])
            d = dr*dr + dg*dg + db*db
            result[i] = int(np.argmin(d))
        return result

# ----------------------------
# ZIP -> 一時ファイル展開（メモリでやるよりシンプルに一時ディレクトリへ）
def unzip_to_temp(zip_path, tmpdir):
    las_files = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith('.las'):
                    out_path = Path(tmpdir) / Path(info.filename).name
                    with open(out_path, 'wb') as f:
                        f.write(zf.read(info.filename))
                    las_files.append(str(out_path))
    except Exception as e:
        logging.exception(f"ZIP 展開に失敗: {zip_path}")
    return las_files

# ----------------------------
# LAS 処理（ファイル単位）。ログとプログレスを受け取る。

def process_las_fast(las_path, output_folder, scale=1, MAX_LINES=10000, use_gpu=False, progress_callback=None, debug=False):
    base_name = Path(las_path).stem
    start_t = time.time()
    logging.info(f"[開始] {las_path}")
    try:
        las = laspy.read(las_path)
    except Exception:
        logging.exception(f"LAS 読み込み失敗: {las_path}")
        raise

    # 座標と色を取得
    X = np.round(las.x / scale).astype(np.int32)
    Y = -np.round(las.y / scale).astype(np.int32)  # Y軸反転（元コード準拠）
    Z = np.round(las.z / scale).astype(np.int32)

    # laspy のフィールド名が異なることがあるため安全に取得
    def safe_get(attr, default=0):
        try:
            return getattr(las, attr)
        except Exception:
            return np.full(X.shape, default, dtype=np.int32)

    R = safe_get('red')
    G = safe_get('green')
    B = safe_get('blue')

    # ユニーク化
    coords = np.stack([X, Y, Z], axis=1)
    coords_unique, idx_unique = np.unique(coords, axis=0, return_index=True)
    X = X[idx_unique]; Y = Y[idx_unique]; Z = Z[idx_unique]
    R = R[idx_unique]; G = G[idx_unique]; B = B[idx_unique]

    n_points = R.shape[0]
    logging.info(f"{base_name}: 点数={n_points}")

    # プログレス: 読込完了
    if progress_callback:
        progress_callback(stage='read', file=las_path, value=1.0)

    # 色マッピング
    indices = None
    try:
        if use_gpu and GPU_AVAILABLE and n_points>10000:
            logging.info("OpenCL GPU を使用して色マッピングを実行します")
            indices = closest_block_opencl(R.astype(np.int32), G.astype(np.int32), B.astype(np.int32))
        else:
            if NUMBA_AVAILABLE:
                logging.info("Numba を使用した CPU 並列マッピングを実行します")
            else:
                logging.info("Numba 未検出。numpy ベースの CPU マッピングを実行します")
            indices = closest_block_cpu(R.astype(np.int32), G.astype(np.int32), B.astype(np.int32))
    except Exception:
        logging.exception("色マッピングで例外が発生しました。CPU フォールバックを試みます")
        # フォールバック
        indices = closest_block_cpu(R.astype(np.int32), G.astype(np.int32), B.astype(np.int32))

    if progress_callback:
        progress_callback(stage='map', file=las_path, value=1.0)

    # 出力書き込み（chunked）
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    line_count = 0
    chunk_index = 1
    out_file = output_folder / f"{base_name}_slice_{chunk_index}.mcfunction"
    try:
        fout = open(out_file, 'w', encoding='utf-8')
    except Exception:
        logging.exception(f"出力ファイルオープン失敗: {out_file}")
        raise

    try:
        for i, (x, y, z, idx) in enumerate(zip(X, Y, Z, indices)):
            fout.write(f"setblock {x} {z} {y} minecraft:{COLOR_MAP[int(idx)][0]}\n")
            line_count += 1
            # 進捗更新: 書き込み比率
            if progress_callback and (i % 5000 == 0):
                progress_callback(stage='write', file=las_path, value=i / max(1, n_points))
            if line_count >= MAX_LINES:
                fout.close()
                chunk_index += 1
                line_count = 0
                out_file = output_folder / f"{base_name}_slice_{chunk_index}.mcfunction"
                fout = open(out_file, 'w', encoding='utf-8')
        if line_count > 0:
            fout.close()
    except Exception:
        logging.exception(f"書き込み中にエラー: {las_path}")
        try:
            fout.close()
        except Exception:
            pass
        raise

    elapsed = time.time() - start_t
    logging.info(f"[完了] {las_path} → {chunk_index} ファイル, 所要 {elapsed:.2f}s")
    if progress_callback:
        progress_callback(stage='done', file=las_path, value=1.0)

# ----------------------------
# プログレス管理ユーティリティ
class ProgressController:
    def __init__(self, total_files, use_rich=USE_RICH, use_tqdm=USE_TQDM):
        self.total_files = total_files
        self.lock = threading.Lock()
        self.file_status = {}
        self.use_rich = use_rich
        self.use_tqdm = use_tqdm
        if self.use_rich:
            self._rich_progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(), TimeRemainingColumn())
            self._rich_task = None
            self._rich_progress.start()
        elif self.use_tqdm:
            self._tqdm = None

    def start_overall(self):
        if self.use_rich:
            self._rich_task = self._rich_progress.add_task("全体", total=self.total_files)
        elif self.use_tqdm:
            from tqdm import tqdm
            self._tqdm = tqdm(total=self.total_files, desc="Files")

    def update_file(self, file, stage, value=0.0):
        with self.lock:
            prev = self.file_status.get(file, 0.0)
            # stage を数値化: read(0.1), map(0.5), write(0.3), done(1.0)
            weight = {'read':0.1, 'map':0.6, 'write':0.2, 'done':1.0}
            newv = value * weight.get(stage, 0.0)
            # 単純に stage 到達で増分
            increment = 0.0
            if stage == 'done':
                increment = 1.0 - prev
                self.file_status[file] = 1.0
            else:
                # 合算していく（過去の値より大きければ更新）
                if newv + prev > prev:
                    increment = max(0.0, min(1.0 - prev, newv))
                    self.file_status[file] = prev + increment
            # 全体バーを更新
            if increment>0:
                if self.use_rich:
                    self._rich_progress.update(self._rich_task, advance=increment)
                elif self.use_tqdm and self._tqdm:
                    self._tqdm.update(increment)
                else:
                    # テキスト表示
                    done = sum(self.file_status.values())
                    pct = (done / self.total_files) * 100
                    print(f"全体進捗: {done:.2f}/{self.total_files} ({pct:.1f}%)", end='\r', flush=True)

    def close(self):
        if self.use_rich:
            self._rich_progress.stop()
        elif self.use_tqdm and self._tqdm:
            self._tqdm.close()

# ----------------------------
# メイン

def find_las_files(input_folder, tmpdir):
    input_path = Path(input_folder)
    las_files = []
    for p in sorted(input_path.iterdir()):
        try:
            if p.suffix.lower() == '.zip':
                las_files.extend(unzip_to_temp(p, tmpdir))
            elif p.suffix.lower() == '.las':
                las_files.append(str(p))
        except Exception:
            logging.exception(f"入力ファイル処理失敗: {p}")
    return las_files


def main():
    parser = argparse.ArgumentParser(description='LAS/ZIP → mcfunction 変換（改良版）')
    parser.add_argument('input_folder')
    parser.add_argument('output_folder')
    parser.add_argument('--threads', '-t', type=int, default=1, help='スレッド数（0=全CPU）')
    parser.add_argument('--gpu', type=int, choices=(0,1), default=0, help='1 で OpenCL GPU を使う（利用可能な場合）')
    parser.add_argument('--max-lines', type=int, default=10000, help='1ファイル当たりの最大行数')
    parser.add_argument('--scale', type=float, default=1.0, help='座標スケール')
    parser.add_argument('--debug', action='store_true', help='デバッグログを有効にする')
    args = parser.parse_args()

    # ログ設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='[%(levelname)s] %(asctime)s %(message)s')

    if args.threads == 0:
        import multiprocessing
        args.threads = multiprocessing.cpu_count()

    if args.gpu and not GPU_AVAILABLE:
        logging.warning("--gpu が指定されましたが OpenCL が使えません。CPU にフォールバックします。")

    # 一時ディレクトリ（ZIP 展開先）
    with tempfile.TemporaryDirectory() as tmpdir:
        las_files = find_las_files(args.input_folder, tmpdir)
        if not las_files:
            logging.error("変換する .las/.zip ファイルが見つかりませんでした。")
            sys.exit(1)

        logging.info(f"処理ファイル数: {len(las_files)}")

        # 進捗コントローラ
        pc = ProgressController(len(las_files))
        pc.start_overall()

        # スレッドセーフにファイルごとに submit する方法（シンプル）
        futures = []
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            for las_file in las_files:
                future = ex.submit(_process_wrapper, las_file, args.output_folder, args.scale, args.max_lines, bool(args.gpu and GPU_AVAILABLE), pc)
                futures.append(future)

            # 完了待ちとエラーハンドリング
            had_error = False
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    had_error = True
                    logging.exception("ファイル処理で例外が発生しました（個別にログ参照）")

        pc.close()

        if had_error:
            logging.warning("一部ファイルでエラーが発生しました。ログを確認してください。")
        else:
            logging.info("すべてのファイルが正常に処理されました。")

# ラッパー: process_las_fast を呼んで進捗コールバックを渡す
def _process_wrapper(las_file, output_folder, scale, max_lines, use_gpu, progress_controller: ProgressController):
    def cb(stage, file, value):
        # コールバックから ProgressController を更新
        progress_controller.update_file(file, stage, value)

    try:
        process_las_fast(las_file, output_folder, scale=scale, MAX_LINES=max_lines, use_gpu=use_gpu, progress_callback=cb)
    except Exception:
        logging.exception(f"処理失敗: {las_file}")
        # 失敗しても ProgressController に done を入れて全体進捗が止まらないようにする
        progress_controller.update_file(las_file, 'done', 1.0)
        raise

if __name__ == '__main__':
    main()
