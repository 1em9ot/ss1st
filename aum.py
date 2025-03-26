# -*- coding: utf-8 -*-

import os
import sys
import csv
import json
import ast
import shutil
import logging
import re
import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
from tkinter.scrolledtext import ScrolledText
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
import subprocess
import threading
import functools
import time
import psutil

###############################################################################
# グローバル定数（修正済み）
###############################################################################
LOG_LEVEL = 'ERROR'
DATA_FOLDER_NAME = "data"
MAX_REBUILD_ATTEMPTS = 3  # 従来は上限付きでしたが、無限リトライに変更します

# 特定のクラスやシンボルを常にリネーム対象から除外するリスト
TOKEN_SAVING_EXCLUDE_LIST = {
    "ConfigManager",
    "CsvPlatformFeatureExtractor",
    "JsonPlatformFeatureExtractor",
    "PlatformSampleManager",
    "PlatformIdentifier",
    # 追加：ファイル名やクラス名自体も変換しない
    "file_identifier",    # モジュール名
    "FileIdentifier",     # クラス名
}

# 特定のモジュールパス(=ファイル)をまるごとリネーム対象から除外する
ABSOLUTE_EXCLUDE_MODULES = {
    "utils/file_identifier.py",
    "parsers/base_parser.py",
}

###############################################################################
# I/Oスロットリング用の定数
###############################################################################
WRITE_RATE_THRESHOLD = 10 * 1024 * 1024  # 10MB/s（しきい値、環境に合わせて調整）
THROTTLE_SLEEP_TIME = 0.5  # 待機時間（秒）

###############################################################################
# アプリ状態管理
###############################################################################
class AUMAppState:
    def __init__(self):
        self.last_build_folder: Optional[str] = None

###############################################################################
# グローバルファイルロック（排他制御用）
###############################################################################
_file_lock_dict: Dict[str, threading.Lock] = {}
_lock_dict_lock = threading.Lock()

def get_file_lock(file_path: str) -> threading.Lock:
    with _lock_dict_lock:
        if file_path not in _file_lock_dict:
            _file_lock_dict[file_path] = threading.Lock()
        return _file_lock_dict[file_path]

def with_file_lock(func):
    @functools.wraps(func)
    def wrapper(file_path, *args, **kwargs):
        lock = get_file_lock(file_path)
        with lock:
            return func(file_path, *args, **kwargs)
    return wrapper

###############################################################################
# リトライ処理の共通ラッパー
###############################################################################
def retry_operation(max_attempts: int = 3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    logging.error(f"{func.__name__} attempt {attempt} failed: {e}")
                    if attempt >= max_attempts:
                        raise
        return wrapper
    return decorator

###############################################################################
# ログ設定
###############################################################################
def setup_logging(level='ERROR'):
    logging.basicConfig(
        filename='aum_error.log',
        level=getattr(logging, level.upper(), logging.ERROR),
        format='%(asctime)s:%(levelname)s:%(message)s',
        encoding='utf-8'
    )

setup_logging(level=LOG_LEVEL)

def log_info(widget: Optional[tk.Text], msg: str):
    logging.info(msg)
    if widget:
        widget.insert("end", msg + "\n")
        widget.see("end")

def log_error(widget: Optional[tk.Text], msg: str):
    logging.error(msg)
    if widget:
        widget.insert("end", "[ERROR] " + msg + "\n")
        widget.see("end")

###############################################################################
# SVN 操作用スケルトン実装（ダミー）
###############################################################################
def svn_commit(repository_path: str, commit_message: str, log_widget=None):
    """
    [ダミー実装]
    指定の SVN リポジトリパスに対して、コミットメッセージ付きで svn commit を実行する関数。
    ※実際の SVN 操作は環境に合わせて実装してください。
    """
    try:
        current_dir = os.getcwd()
        os.chdir(repository_path)
        cmd = ['svn', 'commit', '-m', commit_message]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            log_info(log_widget, f"SVNコミット成功:\n{result.stdout}")
        else:
            log_error(log_widget, f"SVNコミット失敗:\n{result.stderr}")
    except Exception as e:
        log_error(log_widget, f"SVNコミット例外: {e}")
    finally:
        os.chdir(current_dir)

def on_svn_commit(log_widget):
    """
    [ダミー実装]
    GUI 用のハンドラ。ユーザーにリポジトリパスとコミットメッセージを入力させ、
    svn_commit を呼び出します。実際の運用環境に合わせて調整してください。
    """
    repo_path = simpledialog.askstring("SVNリポジトリパス", "SVNのrepositoryパスを入力してください", initialvalue="C:/path/to/svn/repository")
    if not repo_path:
        log_info(log_widget, "SVNリポジトリパスが入力されませんでした。")
        return
    commit_message = simpledialog.askstring("SVNコミット", "コミットメッセージを入力してください")
    if not commit_message:
        log_info(log_widget, "コミットメッセージが入力されませんでした。")
        return
    svn_commit(repo_path, commit_message, log_widget)

###############################################################################
# with_io_throttling: 書き込み前にディスクI/O状況を監視してスロットリングするデコレータ
###############################################################################
def with_io_throttling(func):
    @functools.wraps(func)
    def wrapper(file_path, *args, **kwargs):
        io_before = psutil.disk_io_counters().write_bytes
        time.sleep(THROTTLE_SLEEP_TIME)
        io_after = psutil.disk_io_counters().write_bytes
        write_rate = (io_after - io_before) / THROTTLE_SLEEP_TIME

        if write_rate > WRITE_RATE_THRESHOLD:
            logging.info(f"High disk write rate detected: {write_rate / (1024*1024):.2f} MB/s. Throttling...")
            while write_rate > WRITE_RATE_THRESHOLD:
                time.sleep(THROTTLE_SLEEP_TIME)
                io_before = psutil.disk_io_counters().write_bytes
                time.sleep(THROTTLE_SLEEP_TIME)
                io_after = psutil.disk_io_counters().write_bytes
                write_rate = (io_after - io_before) / THROTTLE_SLEEP_TIME
                logging.info(f"Throttling... current write rate: {write_rate / (1024*1024):.2f} MB/s")
        return func(file_path, *args, **kwargs)
    return wrapper

###############################################################################
# ユーティリティ（ファイルI/O：排他制御＋スロットリング付き）
###############################################################################
@with_file_lock
@with_io_throttling
def read_file_content(file_path: str, log_widget=None) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as fr:
            return fr.read()
    except Exception as e:
        log_error(log_widget, f"ファイル読込失敗: {file_path}: {e}")
    return ""

@with_file_lock
@with_io_throttling
def write_file_content(file_path: str, content: str, log_widget=None):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as fw:
            fw.write(content)
    except Exception as e:
        log_error(log_widget, f"ファイル書込み失敗: {file_path}: {e}")

###############################################################################
# generate_summary: ファイル種別に応じたサマリー生成
###############################################################################
def generate_summary(file_path: str, content: str) -> str:
    if file_path.endswith('.py'):
        try:
            tree = ast.parse(content)
            funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            clss = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            docstrings = ast.get_docstring(tree)
            s = "Pythonコード。\n"
            if clss:
                s += f"クラス: {', '.join(clss)}。\n"
            if funcs:
                s += f"関数: {', '.join(funcs)}。\n"
            if docstrings:
                s += f"Docstring(冒頭60文字): {docstrings[:60]}...\n"
            return s
        except Exception as e:
            logging.error(f"AST解析失敗: {file_path}: {e}", exc_info=True)
            return f"解析エラー: {e}\n"
    elif file_path.endswith('.ipynb'):
        try:
            nb = json.loads(content)
            cells = nb.get('cells', [])
            m = sum(1 for c in cells if c.get('cell_type') == 'markdown')
            c = sum(1 for c in cells if c.get('cell_type') == 'code')
            return f"Jupyter Notebook。\nセル合計={len(cells)}, Markdown={m}, Code={c}\n"
        except Exception as e:
            return f"Jupyter Notebook。\nNotebook解析エラー: {e}\n"
    elif file_path.endswith('.json'):
        try:
            obj = json.loads(content)
            if isinstance(obj, dict):
                keys = list(obj.keys())
                keys_display = ", ".join(keys[:10])
                if len(keys) > 10:
                    keys_display += "..."
                return f"JSONファイル (辞書)。キー: {keys_display}\n"
            elif isinstance(obj, list):
                return f"JSONファイル (リスト)。要素数: {len(obj)}\n"
            else:
                return "JSONファイル (その他形式)。\n"
        except Exception as e:
            return f"JSON解析エラー: {e}\n"
    elif file_path.endswith('.tex'):
        lines = content.count('\n')
        return f"TeXファイル。行数: {lines}\n"
    else:
        lines = content.count('\n')
        return f"その他テキスト。\n行数: {lines}\n"

###############################################################################
# name_map.csv の衝突修正
###############################################################################
def detect_symbol_collisions(mapping: Dict[str, str]) -> set:
    from collections import defaultdict
    rev = defaultdict(list)
    for orig, short_ in mapping.items():
        rev[short_].append(orig)
    return {k for k, v in rev.items() if len(v) > 1}

def auto_rename_collisions_in_name_map(csv_path: str, log_widget=None) -> bool:
    if not os.path.exists(csv_path):
        return False
    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            for r in rd:
                rows.append(r)
    except Exception as e:
        log_error(log_widget, f"name_map.csv 読込失敗: {e}")
        return False
    mapping = {r['original_name']: r['short_name'] for r in rows}
    collisions = detect_symbol_collisions(mapping)
    if not collisions:
        return False
    short_name_counter = {}
    renamed_count = 0
    for r in rows:
        sh = r['short_name']
        if sh in collisions:
            base = sh
            short_name_counter.setdefault(base, 0)
            short_name_counter[base] += 1
            r['short_name'] = base + str(short_name_counter[base])
            renamed_count += 1
    if renamed_count:
        try:
            with open(csv_path, 'w', encoding='utf-8', newline='') as fw:
                w = csv.writer(fw)
                w.writerow(["original_name", "short_name"])
                for row in rows:
                    w.writerow([row['original_name'], row['short_name']])
            log_info(log_widget, f"name_map.csv 衝突を {renamed_count} 件リネーム修正")
            return True
        except Exception as e:
            log_error(log_widget, f"name_map.csv 修正書込失敗: {e}")
    return False

###############################################################################
# collected_scripts.json 修復
###############################################################################
def auto_repair_collected_scripts(base_dir: str, log_widget) -> None:
    cjson_path = os.path.join(base_dir, "collected_scripts.json")
    if not os.path.exists(cjson_path):
        return
    try:
        with open(cjson_path, 'r', encoding='utf-8') as fr:
            content = fr.read().strip()
            if not content:
                raise ValueError("Empty file")
            data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("トップレベルがdictではない")
        if "scripts" in data and not isinstance(data["scripts"], list):
            raise ValueError("'scripts'がlistではない")
    except Exception as e:
        log_error(log_widget, f"collected_scripts.json 破損: {e}")
        backup = cjson_path + ".bak"
        try:
            shutil.move(cjson_path, backup)
            log_info(log_widget, f"破損ファイルをバックアップ→ {backup}")
        except Exception as e_backup:
            log_error(log_widget, f"バックアップ失敗: {e_backup}")
        data = {
            "system_overview": "このシステムは...（必要に応じて書き換え）",
            "settings": {},
            "scripts": []
        }
        try:
            with open(cjson_path, 'w', encoding='utf-8') as fw:
                json.dump(data, fw, ensure_ascii=False, indent=2)
            log_info(log_widget, "collected_scripts.json を新規作成")
        except Exception as e_write:
            log_error(log_widget, f"新規作成失敗: {e_write}")

###############################################################################
# 三方比較マージ
###############################################################################
def merge_collected_scripts(original: dict, new: dict, file_data: Dict[str, dict], log_widget) -> dict:
    _merge_system_and_settings(original, new)
    old_scripts = original.get("scripts", [])
    old_map = {s['path']: s for s in old_scripts if 'path' in s}
    new_scripts = new.get("scripts", [])
    new_map = {s.get('path'): s for s in new_scripts if s.get('path')}
    for p, new_sc in new_map.items():
        if p not in old_map:
            old_map[p] = new_sc
            log_info(log_widget, f"新規追加: {p}")
        else:
            old_sc = old_map[p]
            _merge_single_script(old_sc, new_sc, file_data, p, log_widget)
    merged_list = list(old_map.values())
    original["scripts"] = merged_list
    return original

def _merge_system_and_settings(old: dict, new: dict):
    if new.get("system_overview"):
        old["system_overview"] = new["system_overview"]
    if "settings" in new and isinstance(new["settings"], dict):
        old.setdefault("settings", {}).update(new["settings"])

def _merge_single_script(old_sc: dict, new_sc: dict, file_data: Dict[str, dict], p: str, log_widget):
    old_cont = old_sc.get("content", "")
    new_cont = new_sc.get("content", "")
    disk_cont = file_data.get(p, {}).get("content", "")
    conflict = (old_cont != disk_cont and new_cont != disk_cont and old_cont != new_cont)
    if conflict:
        log_info(log_widget, f"衝突あり: {p}")
    else:
        for k, v in new_sc.items():
            if k != "path":
                old_sc[k] = v
        log_info(log_widget, f"上書き: {p}")

###############################################################################
# __init__.py 自動生成
###############################################################################
def ensure_init_py_for_python_dirs(base_dir: str, log_widget):
    exclude = {'venv', '__pycache__', '.git', 'outputs', 'temp_build', 'backups', DATA_FOLDER_NAME}
    py_dirs = set()
    for root, dirs, files in os.walk(base_dir):
        if any(f.endswith('.py') for f in files):
            py_dirs.add(root)
    for d in py_dirs:
        init_file = os.path.join(d, "__init__.py")
        if not os.path.isfile(init_file):
            write_file_content(init_file, "", log_widget)
            log_info(log_widget, f"__init__.py 自動生成: {os.path.relpath(init_file, base_dir)}")

###############################################################################
# config.json => collected_scripts.json の更新（設定取り込み）
###############################################################################
def restore_settings_from_config_json(base_dir: str, log_widget):
    """
    your_project直下または your_project/your_project にある config.json を探し、
    設定情報を collected_scripts.json の settings に反映する。
    ※設定ファイル内に "settings" キーがあれば、その中身を使用（そうでなければファイル全体を設定として取り込む）
    """
    possible_cfg_paths = [
        os.path.join(base_dir, "your_project", "config.json"),
        os.path.join(base_dir, "your_project", "your_project", "config.json"),
    ]

    cfg_path = None
    for pth in possible_cfg_paths:
        if os.path.isfile(pth):
            cfg_path = pth
            break

    if not cfg_path:
        log_info(log_widget, "config.jsonが見つからないためスキップ")
        return

    cjson_path = os.path.join(base_dir, "collected_scripts.json")
    if not os.path.isfile(cjson_path):
        log_info(log_widget, f"collected_scripts.jsonが無い: {cjson_path}")
        return

    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg_data = json.load(f)
    except Exception as e:
        log_error(log_widget, f"config.json 読込失敗: {e}")
        return

    if not isinstance(cfg_data, dict):
        log_info(log_widget, "config.jsonの形式が不正です。")
        return

    # 設定ファイル内に "settings" キーがあればその中身を使用（なければ全体を設定として取り込む）
    new_settings = cfg_data.get("settings", cfg_data)

    try:
        with open(cjson_path, 'r', encoding='utf-8') as fr:
            cjson_data = json.load(fr)
        if not isinstance(cjson_data, dict):
            cjson_data = {"system_overview": "", "settings": {}, "scripts": []}
    except Exception as e:
        log_error(log_widget, f"collected_scripts.json 読込失敗: {e}")
        return

    # 既存の設定に新しい設定をマージする
    cjson_data.setdefault("settings", {}).update(new_settings)
    try:
        with open(cjson_path, 'w', encoding='utf-8') as fw:
            json.dump(cjson_data, fw, ensure_ascii=False, indent=2)
        log_info(log_widget, f"{cjson_path} の settings を更新しました")
    except Exception as e:
        log_error(log_widget, f"config→collected_scripts 反映失敗: {e}")

###############################################################################
# コード収集
###############################################################################
def collect(base_dir: str, log_widget):
    log_info(log_widget, "コード収集を開始...")
    auto_repair_collected_scripts(base_dir, log_widget)
    output_folder = os.path.join(base_dir, "your_project")
    cjson = os.path.join(base_dir, "collected_scripts.json")
    if os.path.isfile(cjson):
        try:
            with open(cjson, 'r', encoding='utf-8') as fr:
                old_data = json.load(fr)
            if not isinstance(old_data, dict):
                old_data = {"system_overview": "", "settings": {}, "scripts": []}
        except:
            old_data = {"system_overview": "", "settings": {}, "scripts": []}
    else:
        old_data = {"system_overview": "", "settings": {}, "scripts": []}

    ensure_init_py_for_python_dirs(base_dir, log_widget)

    target_ext = ('.py', '.json', '.tex')
    exclude_dirs = {'venv', '__pycache__', '.git', 'outputs', 'temp_build', 'backups', DATA_FOLDER_NAME}
    exclude_files = {'collected_scripts.json', os.path.basename(__file__)}
    new_list = []
    for root, dirs, files in os.walk(base_dir):
        abs_root = os.path.abspath(root)
        abs_output = os.path.abspath(output_folder)
        if abs_root.startswith(abs_output) and "your_project" not in os.path.relpath(abs_root, base_dir).split(os.sep)[0]:
            continue
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fn in files:
            if fn in exclude_files:
                continue
            if any(fn.endswith(e) for e in target_ext):
                fp = os.path.join(root, fn)
                if fp == os.path.abspath(__file__):
                    continue
                content = read_file_content(fp, log_widget)
                if not content:
                    continue
                relp = os.path.relpath(fp, base_dir)
                summ = generate_summary(fp, content)
                new_list.append({"path": relp, "overview": summ, "content": content})
                log_info(log_widget, f"収集: {relp}")

    file_map = {s["path"]: {"content": s["content"], "overview": s["overview"]} for s in new_list}
    merged = merge_collected_scripts(
        old_data,
        {
            "system_overview": old_data.get("system_overview", ""),
            "settings": old_data.get("settings", {}),
            "scripts": new_list
        },
        file_map,
        log_widget
    )
    try:
        with open(cjson, 'w', encoding='utf-8') as fw:
            json.dump(merged, fw, ensure_ascii=False, indent=2)
        log_info(log_widget, f"出力完了: {cjson}")
    except Exception as e:
        log_error(log_widget, f"書き込み失敗: {e}")

    restore_settings_from_config_json(base_dir, log_widget)

###############################################################################
# Pythonシンボル名 短縮
###############################################################################
def collect_python_symbols_for_map(base_dir: str) -> set:
    import ast
    exclude = {'venv', '__pycache__', '.git', 'temp_build', 'backups', DATA_FOLDER_NAME}
    found = set()
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root).lower() == DATA_FOLDER_NAME.lower():
            continue
        abs_root = os.path.abspath(root)
        abs_output = os.path.abspath(os.path.join(base_dir, "your_project"))
        if abs_root.startswith(abs_output):
            continue
        dirs[:] = [d for d in dirs if d not in exclude]
        for f in files:
            if f.endswith('.py') and f != os.path.basename(__file__):
                fp = os.path.join(root, f)
                txt = read_file_content(fp)
                if not txt:
                    continue
                try:
                    tree = ast.parse(txt)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            found.add(node.name)
                        elif isinstance(node, ast.ClassDef):
                            found.add(node.name)
                except:
                    pass
    return found

def generate_short_symbol_name(orig_name: str) -> str:
    if not orig_name:
        return "x"
    return orig_name[0].lower() + "_sym"

def ensure_name_map_csv_exists(base_dir: str) -> str:
    csv_path = os.path.join(base_dir, "name_map.csv")
    syms = collect_python_symbols_for_map(base_dir)
    existing_map = {}
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as fr:
                rd = csv.DictReader(fr)
                for r in rd:
                    o = r['original_name']
                    s = r['short_name']
                    existing_map[o] = s
        except:
            pass
    else:
        logging.info("name_map.csvがないため新規作成")
    for s in syms:
        if s in TOKEN_SAVING_EXCLUDE_LIST:
            existing_map[s] = s
        else:
            if s not in existing_map:
                existing_map[s] = generate_short_symbol_name(s)
    with open(csv_path, 'w', encoding='utf-8', newline='') as fw:
        w = csv.writer(fw)
        w.writerow(["original_name", "short_name"])
        for k, v in sorted(existing_map.items()):
            w.writerow([k, v])
    return csv_path

def load_name_mappings(csv_path: str) -> dict:
    mp = {}
    if not os.path.isfile(csv_path):
        return mp
    try:
        with open(csv_path, 'r', encoding='utf-8') as fr:
            rd = csv.DictReader(fr)
            for r in rd:
                mp[r['original_name']] = r['short_name']
    except:
        pass
    return mp

def invert_name_map(mp: dict) -> dict:
    return {v: k for k, v in mp.items()}

###############################################################################
# AST変換 (識別子置換)
###############################################################################
GLOBAL_EXCLUDE_MODULES = ABSOLUTE_EXCLUDE_MODULES

def apply_name_mappings_ast(content: str, name_map: dict, path_for_check: str) -> str:
    norm_path = os.path.normpath(path_for_check).replace("\\", "/")
    for exc_subpath in GLOBAL_EXCLUDE_MODULES:
        if exc_subpath in norm_path:
            return content

    FIXED_EXCLUDE = TOKEN_SAVING_EXCLUDE_LIST.copy()
    current_exclude = set()
    attempt = 0

    def extract_problematic_identifier(e: Exception) -> Optional[str]:
        msg = str(e)
        m = re.search(r"object has no attribute '(\w+)'", msg)
        if m:
            return m.group(1)
        m = re.search(r"cannot import name '(\w+)'", msg)
        if m:
            return m.group(1)
        return None

    class ImportCollector(ast.NodeVisitor):
        def __init__(self):
            self.imported = set()
        def visit_Import(self, node):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split('.')[0]
                self.imported.add(name)
            self.generic_visit(node)
        def visit_ImportFrom(self, node):
            if node.module:
                self.imported.add(node.module.split('.')[0])
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                self.imported.add(name)
            self.generic_visit(node)

    class Mapper(ast.NodeTransformer):
        def __init__(self, exclude_set, name_map):
            self.exclude = exclude_set
            self.nmap = name_map
            super().__init__()
        def visit_Name(self, node):
            if node.id.startswith("__") and node.id.endswith("__"):
                return node
            if node.id in self.exclude:
                return node
            if node.id in self.nmap:
                node.id = self.nmap[node.id]
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            if not (node.name.startswith("__") and node.name.endswith("__")):
                if node.name not in self.exclude and node.name in self.nmap:
                    node.name = self.nmap[node.name]
            self.generic_visit(node)
            return node
        def visit_ClassDef(self, node):
            if not (node.name.startswith("__") and node.name.endswith("__")):
                if node.name not in self.exclude and node.name in self.nmap:
                    node.name = self.nmap[node.name]
            self.generic_visit(node)
            return node
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                return node
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return node
            if node.attr in self.exclude:
                return node
            if node.attr in self.nmap:
                node.attr = self.nmap[node.attr]
            return node
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name in self.exclude:
                    continue
                if alias.name in self.nmap:
                    alias.name = self.nmap[alias.name]
                if alias.asname and alias.asname in self.nmap:
                    alias.asname = self.nmap[alias.asname]
            return node
        def visit_ImportFrom(self, node):
            if node.module in self.exclude:
                return node
            if node.module and (node.module in self.nmap):
                node.module = self.nmap[node.module]
            for alias in node.names:
                if alias.name in self.exclude:
                    continue
                if alias.name in self.nmap:
                    alias.name = self.nmap[alias.name]
                if alias.asname and alias.asname in self.nmap:
                    alias.asname = self.nmap[alias.asname]
            return node

    while True:
        attempt += 1
        try:
            tree = ast.parse(content)
            collector = ImportCollector()
            collector.visit(tree)
            auto_exclude = collector.imported
            total_exclude = FIXED_EXCLUDE.union(auto_exclude).union(current_exclude)
            mapper = Mapper(total_exclude, name_map)
            new_tree = mapper.visit(tree)
            ast.fix_missing_locations(new_tree)
            if hasattr(ast, 'unparse'):
                out_code = ast.unparse(new_tree)
            else:
                import astor
                out_code = astor.to_source(new_tree)
            return out_code
        except Exception as e:
            problematic = extract_problematic_identifier(e)
            if problematic:
                if problematic not in current_exclude:
                    current_exclude.add(problematic)
                else:
                    return content
            else:
                return content

###############################################################################
# 依存解析 & トポロジカルソート
###############################################################################
def analyze_dependencies_and_sort(scripts: List[Dict[str, str]]) -> List[Dict[str, str]]:
    graph = {}
    script_map = {}
    for sc in scripts:
        p = sc["path"]
        script_map[p] = sc
        graph[p] = set()

    path_by_modname = {}
    for sc in scripts:
        p = sc["path"].replace("\\", "/")
        if p.endswith(".py"):
            mod_parts = p.split("/")
            if mod_parts[0] == "your_project":
                mod_parts = mod_parts[1:]
            py_base = mod_parts[-1][:-3]
            mod_parts[-1] = py_base
            dotted = ".".join(mod_parts)
            path_by_modname[dotted] = p

    def extract_imports(p: str, code: str) -> Set[str]:
        import_list = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod = alias.name.split(".")[0]
                        import_list.add(mod)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        base = node.module.split(".")[0]
                        import_list.add(base)
        except:
            pass
        return import_list

    for sc in scripts:
        p = sc["path"]
        code = sc["content"]
        imported = extract_imports(p, code)
        for imp in imported:
            for k, v in path_by_modname.items():
                if k.startswith(imp):
                    graph[p].add(v)

    in_degree = {p: 0 for p in graph}
    for p in graph:
        for dep in graph[p]:
            if dep in in_degree:
                in_degree[dep] += 1

    from collections import deque
    queue = deque([p for p in in_degree if in_degree[p] == 0])
    sorted_paths = []
    while queue:
        u = queue.popleft()
        sorted_paths.append(u)
        for dep in graph[u]:
            if dep in in_degree:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

    if len(sorted_paths) < len(graph):
        logging.error("循環依存関係が検出されました。依存順序の保証ができません。")
        return scripts

    path_to_index = {p: i for i, p in enumerate(sorted_paths)}
    scripts_sorted = sorted(scripts, key=lambda sc: path_to_index.get(sc["path"], 999999))
    return scripts_sorted

###############################################################################
# ビルド (順序づけ + AST変換 + 出力)
###############################################################################
def build_in_topological_order(
    scripts: List[Dict[str, str]], 
    out_dir: str, 
    name_map: Dict[str, str],
    log_widget
) -> Tuple[int, List[str]]:
    sorted_scripts = analyze_dependencies_and_sort(scripts)
    logs = []
    built_count = 0
    for sc in sorted_scripts:
        path_ = sc["path"]
        content_ = sc["content"]
        if path_.endswith(".py"):
            new_content = apply_name_mappings_ast(content_, name_map, path_for_check=path_)
        elif path_.endswith(".json") or path_.endswith(".tex"):
            new_content = content_
        else:
            continue
        rel_path = path_
        if rel_path.startswith("your_project" + os.path.sep):
            rel_path = rel_path[len("your_project" + os.path.sep):]
        target_file = os.path.join(out_dir, rel_path)
        try:
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            with open(target_file, 'w', encoding='utf-8') as fw:
                fw.write(new_content)
            logs.append(f"出力: {os.path.relpath(target_file, out_dir)}")
            built_count += 1
        except Exception as e:
            logs.append(f"書き込みエラー: {target_file}: {e}")
    return built_count, logs

def verify_build_results(build_dir: str, collected_json: str, log_widget) -> bool:
    try:
        with open(collected_json, 'r', encoding='utf-8') as fr:
            data = json.load(fr)
    except Exception as e:
        log_error(log_widget, f"収集結果の読込失敗: {e}")
        return False
    scripts = data.get("scripts", [])
    all_ok = True
    for script in scripts:
        p = script.get("path", "")
        if not (p.endswith(".py") or p.endswith(".json") or p.endswith(".tex")):
            continue
        rel_path = p
        if rel_path.startswith("your_project" + os.path.sep):
            rel_path = rel_path[len("your_project" + os.path.sep):]
        build_file = os.path.join(build_dir, rel_path)
        if not os.path.exists(build_file):
            log_error(log_widget, f"ビルド結果に存在しない: {build_file}")
            all_ok = False
            continue
        with open(build_file, 'r', encoding='utf-8') as bf:
            build_content = bf.read()
        if build_content.strip() != script.get("content", "").strip():
            log_error(log_widget, f"内容不一致: {build_file}")
            all_ok = False
        else:
            log_info(log_widget, f"チェックOK: {build_file}")
    return all_ok

def _build_config_json(settings: dict, out_dir: str, log_widget):
    """
    修正ポイント:
    出力する config.json の内容をトップレベルで "settings" キーを持つ形に変更。
    """
    cfg_path = os.path.join(out_dir, "config.json")
    try:
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        with open(cfg_path, 'w', encoding='utf-8') as fw:
            json.dump({"settings": settings}, fw, ensure_ascii=False, indent=2)
        log_info(log_widget, f"config.json生成: {cfg_path}")
    except Exception as e:
        log_error(log_widget, f"config.json 書込失敗: {e}")

def build_project(
    base_dir: str,
    log_widget,
    is_obfuscate: bool = True
) -> Optional[str]:
    auto_repair_collected_scripts(base_dir, log_widget)
    cjson = os.path.join(base_dir, "collected_scripts.json")
    if not os.path.isfile(cjson):
        messagebox.showerror("エラー", f"{cjson} が見つかりません")
        return None
    csv_path = ensure_name_map_csv_exists(base_dir)
    direct_map = load_name_mappings(csv_path)
    final_map = direct_map if is_obfuscate else invert_name_map(direct_map)
    tmp_build_root = os.path.join(base_dir, "temp_build")
    os.makedirs(tmp_build_root, exist_ok=True)
    out_dir = os.path.join(tmp_build_root, "your_project_temp")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_info(log_widget, f"一時出力先: {out_dir}")
    with open(cjson, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    scripts = data.get("scripts", [])
    settings = data.get("settings", {})
    py_scripts = [s for s in scripts if s.get("path", "").endswith((".py", ".json", ".tex"))]
    built_count, logs_texts = build_in_topological_order(py_scripts, out_dir, final_map, log_widget)
    _build_config_json(settings, out_dir, log_widget)
    built_count += 1
    if logs_texts:
        for l in logs_texts:
            log_info(log_widget, l)
    log_info(log_widget, f"ビルド完了: {built_count}ファイル")
    if not verify_build_results(out_dir, cjson, log_widget):
        log_error(log_widget, "収集結果とビルド結果が一致しません")
    else:
        log_info(log_widget, "ビルド結果と収集結果は一致しています")
    return out_dir

def finalize_build(temp_out_dir: str, base_dir: str, log_widget, app_state: AUMAppState):
    if not temp_out_dir or not os.path.isdir(temp_out_dir):
        log_error(log_widget, f"一時フォルダが無い: {temp_out_dir}")
        return
    final_dir = os.path.join(base_dir, "your_project")
    if os.path.isdir(final_dir):
        backups_dir = os.path.join(base_dir, "backups")
        os.makedirs(backups_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = os.path.join(backups_dir, f"your_project_old_{stamp}")
        try:
            shutil.make_archive(zip_name, 'zip', final_dir)
            log_info(log_widget, f"既存フォルダをzipバックアップ: {zip_name}.zip")
        except Exception as e:
            log_error(log_widget, f"バックアップ失敗: {e}")
        try:
            shutil.rmtree(final_dir)
            log_info(log_widget, f"既存フォルダ削除: {final_dir}")
        except Exception as e:
            log_error(log_widget, f"削除失敗: {e}")
            return
    try:
        shutil.move(temp_out_dir, final_dir)
        log_info(log_widget, f"上書き完了: {final_dir}")
        app_state.last_build_folder = base_dir
    except Exception as e:
        log_error(log_widget, f"移動失敗: {e}")
        return
    tmpb = os.path.join(base_dir, "temp_build")
    if os.path.isdir(tmpb):
        try:
            shutil.rmtree(tmpb)
            log_info(log_widget, f"temp_build削除: {tmpb}")
        except Exception as ex:
            log_error(log_widget, f"temp_build削除失敗: {ex}")
    log_info(log_widget, f"last_build_folder={app_state.last_build_folder}")

###############################################################################
# トークン節約ビルド（無限リトライ版）
###############################################################################
def token_saving_build_unified(base_dir: str, log_widget, app_state: AUMAppState):
    tries = 0
    while True:  # 無限ループで再試行
        tries += 1
        log_info(log_widget, f"ビルド試行 {tries} 回目")
        tmp_out = build_project(base_dir, log_widget, is_obfuscate=True)
        if not tmp_out:
            log_error(log_widget, "build_project で出力が得られませんでした。")
            continue
        finalize_build(tmp_out, base_dir, log_widget, app_state)
        run_main_py(base_dir, log_widget, app_state)
        all_logs = log_widget.get("1.0", "end")
        if ("ImportError:" in all_logs) or ("NameError:" in all_logs) or ("ModuleNotFoundError:" in all_logs):
            log_info(log_widget, "エラー検知（名前衝突等）。name_map.csv を修正して再試行します。")
            csvp = os.path.join(base_dir, "name_map.csv")
            fixed = auto_rename_collisions_in_name_map(csvp, log_widget)
            if fixed:
                continue
            else:
                log_error(log_widget, "name_map.csv の自動修正に失敗しました。再試行します。")
                continue
        else:
            log_info(log_widget, "ビルド成功：エラーなし。")
            break

def restore_build(base_dir: str, log_widget, app_state: AUMAppState):
    tmp_out = build_project(base_dir, log_widget, is_obfuscate=False)
    if tmp_out:
        finalize_build(tmp_out, base_dir, log_widget, app_state)

def run_main_py(base_dir: str, log_widget, app_state: AUMAppState):
    final_dir = os.path.join(base_dir, "your_project")
    main_py = os.path.join(final_dir, "main.py")
    data_path = os.path.join(final_dir, DATA_FOLDER_NAME)
    log_info(log_widget, f"main.py => {main_py}")
    log_info(log_widget, f"--input => {data_path}")
    if not os.path.isfile(main_py):
        log_error(log_widget, "main.py が見つからない")
        return
    if not os.path.isdir(data_path):
        log_info(log_widget, f"dataフォルダなし: {data_path}")
    cmd = [sys.executable, main_py, "--input", data_path, "--skip-parent-dir"]
    try:
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode != 0:
            log_error(log_widget, f"エラー({ret.returncode}): {ret.stderr}")
        else:
            log_info(log_widget, f"実行完了:\n{ret.stdout}")
    except Exception as e:
        log_error(log_widget, f"例外: {e}")
    gather_run_artifacts(base_dir, log_widget, app_state)
    generate_summary_report(base_dir, log_widget)

def gather_run_artifacts(base_dir: str, log_widget, app_state: AUMAppState):
    final_dir = os.path.join(base_dir, "your_project")
    if not os.path.isdir(final_dir):
        log_info(log_widget, "your_projectがない")
        return
    runs = []
    for d in os.listdir(final_dir):
        if d.startswith("run_"):
            fp = os.path.join(final_dir, d)
            if os.path.isdir(fp):
                runs.append(fp)
    if not runs:
        log_info(log_widget, "run_*** フォルダなし")
        return
    runs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest = runs[0]
    log_info(log_widget, f"最新run: {latest}")
    fb_src = os.path.join(latest, "feedback.json")
    fb_dst = os.path.join(base_dir, "feedback.json")
    if os.path.isfile(fb_src):
        shutil.copy2(fb_src, fb_dst)
        log_info(log_widget, f"feedback.json-> {fb_dst}")
    logs_src = os.path.join(latest, "logs")
    logs_dst = os.path.join(base_dir, "logs")
    if os.path.isdir(logs_src):
        if os.path.isdir(logs_dst):
            shutil.rmtree(logs_dst)
        shutil.copytree(logs_src, logs_dst)
        log_info(log_widget, f"logs-> {logs_dst}")
    else:
        log_info(log_widget, "logsなし")

def generate_summary_report(base_dir: str, log_widget):
    log_info(log_widget, "レポート生成...")
    fb = os.path.join(base_dir, "feedback.json")
    logsdir = os.path.join(base_dir, "logs")
    outp = os.path.join(base_dir, "summary_report.md")
    data = {}
    if os.path.isfile(fb):
        try:
            with open(fb, 'r', encoding='utf-8') as fr:
                data = json.load(fr)
        except:
            pass
    tstamp = data.get("timestamp", "N/A")
    st = data.get("status", "N/A")
    remarks = data.get("remarks", "")
    improvs = data.get("possible_improvements", [])
    logs_sum = []
    if os.path.isdir(logsdir):
        for root, dirs, files in os.walk(logsdir):
            for f in files:
                if f.endswith(".log"):
                    fp = os.path.join(root, f)
                    try:
                        with open(fp, 'r', encoding='utf-8', errors='replace') as ff:
                            lines = ff.readlines()
                        errwarn = [ln.strip() for ln in lines if ("ERROR" in ln or "WARN" in ln)]
                        relp = os.path.relpath(fp, logsdir)
                        logs_sum.append((relp, errwarn))
                    except:
                        pass
    try:
        with open(outp, 'w', encoding='utf-8') as fw:
            fw.write("# Analysis Summary Report\n\n")
            fw.write(f"- **Timestamp**: {tstamp}\n")
            fw.write(f"- **Status**: {st}\n")
            fw.write(f"- **Remarks**: {remarks}\n\n")
            fw.write("## Possible Improvements\n")
            if improvs:
                for x in improvs:
                    fw.write(f"- {x}\n")
            else:
                fw.write("- (No improvements)\n")
            fw.write("\n## Logs Summary (ERROR/WARN lines)\n")
            if logs_sum:
                for (fn, lines) in logs_sum:
                    fw.write(f"### {fn}\n")
                    if lines:
                        for ln in lines:
                            fw.write(f"- {ln}\n")
                    else:
                        fw.write("- (No ERROR/WARN)\n")
            else:
                fw.write("(No logs found or no ERROR/WARN lines)\n")
        log_info(log_widget, f"summary_report.md 生成: {outp}")
    except Exception as e:
        log_error(log_widget, f"レポート生成失敗: {e}")

###############################################################################
# 新関数: central_merge（中央集約型マージ処理）
###############################################################################
def central_merge(old_data: dict, new_data: dict, file_map: Dict[str, dict], log_widget) -> dict:
    """
    複数プロセスからのマージ要求をシリアルに処理するための関数。
    入力:
      - old_data: 既存の collected_scripts.json の内容
      - new_data: 新たに収集されたデータ
      - file_map: 各ファイルの内容を保持した dict（キー: ファイルパス）
      - log_widget: ログ出力用ウィジェット
    出力:
      - マージ後の dict
    """
    return merge_collected_scripts(old_data, new_data, file_map, log_widget)

###############################################################################
# GUI用補助関数: patch_ui, clean, enhanced_smart_paste
###############################################################################
def patch_ui(parent_frame: tk.Frame, base_dir: str, log_widget):
    frm = tk.LabelFrame(parent_frame, text="パッチ適用(文字列置換)")
    frm.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    tk.Label(frm, text="ターゲット:").pack(side=tk.LEFT)
    e_tgt = tk.Entry(frm, width=40)
    e_tgt.pack(side=tk.LEFT, padx=5)
    def on_browse():
        path = filedialog.askopenfilename(
            initialdir=base_dir,
            filetypes=[("Python", ".py"), ("All Files", "*.*")]
        )
        if path:
            e_tgt.delete(0, tk.END)
            e_tgt.insert(0, path)
    tk.Button(frm, text="参照", command=on_browse).pack(side=tk.LEFT, padx=5)
    tk.Label(frm, text="検索:").pack(side=tk.LEFT)
    e_search = tk.Entry(frm, width=15)
    e_search.pack(side=tk.LEFT, padx=2)
    tk.Label(frm, text="置換:").pack(side=tk.LEFT)
    e_replace = tk.Entry(frm, width=15)
    e_replace.pack(side=tk.LEFT, padx=2)
    def do_patch():
        tf = e_tgt.get().strip()
        st = e_search.get()
        rp = e_replace.get()
        if not os.path.isfile(tf):
            log_error(log_widget, f"ファイルが無い: {tf}")
            return
        if not st:
            log_info(log_widget, "検索文字が空")
            return
        try:
            with open(tf, 'r', encoding='utf-8') as fr:
                txt = fr.read()
            new_txt = txt.replace(st, rp)
            if new_txt == txt:
                log_info(log_widget, f"'{st}' が見つからず")
            else:
                with open(tf, 'w', encoding='utf-8') as fw:
                    fw.write(new_txt)
                log_info(log_widget, f"パッチ適用完了: {tf}")
        except Exception as e:
            log_error(log_widget, f"パッチエラー: {e}")
    tk.Button(frm, text="パッチ適用", command=do_patch).pack(side=tk.LEFT, padx=5)

def clean(base_dir: str, log_widget):
    log_info(log_widget, "[Clean] 開始")
    keep = {
        os.path.abspath(__file__),
        os.path.join(base_dir, 'collected_scripts.json'),
        os.path.join(base_dir, 'name_map.csv')
    }
    todel = []
    for x in os.listdir(base_dir):
        p = os.path.join(base_dir, x)
        if os.path.abspath(p) in keep:
            continue
        todel.append(p)
    if not todel:
        messagebox.showinfo("情報", "削除対象なし")
        return
    r = messagebox.askyesno("確認", "削除しますか？\n\n" + "\n".join(todel))
    if r:
        for item in todel:
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                log_info(log_widget, f"削除: {item}")
            except Exception as e:
                log_error(log_widget, f"削除失敗: {item}:{e}")
        log_info(log_widget, "完了")
    else:
        log_info(log_widget, "キャンセル")

def enhanced_smart_paste(base_dir: str, log_widget):
    txt = simpledialog.askstring("Enhanced Smart Paste", "ChatGPT等の出力をペースト")
    if not txt:
        log_info(log_widget, "キャンセルされました")
        return
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bkup = os.path.join(base_dir, f"pasted_backup_{ts}.txt")
    write_file_content(bkup, txt, log_widget)
    log_info(log_widget, f"全文バックアップ:{bkup}")
    try:
        jobj = json.loads(txt)
        log_info(log_widget, "全体をJSONマージ試行")
        ok = try_merge_into_collected(jobj, base_dir, log_widget)
        if ok:
            csvp = os.path.join(base_dir, 'name_map.csv')
            auto_rename_collisions_in_name_map(csvp, log_widget)
            log_info(log_widget, "衝突修正完了")
            return
    except:
        pass
    blocks = extract_code_blocks_and_text(txt)
    for bt, body, lang in blocks:
        if bt == "code":
            process_code_block(body, lang, base_dir, log_widget)
        else:
            process_text_block(body, base_dir, log_widget)
    csvp = os.path.join(base_dir, 'name_map.csv')
    auto_rename_collisions_in_name_map(csvp, log_widget)
    log_info(log_widget, "最終衝突修正完了")

def extract_code_blocks_and_text(full_text: str) -> List[Tuple[str, str, str]]:
    CODE_BLOCK_PATTERN = re.compile(r'```(\w+)?(.*?)```', re.DOTALL)
    res = []
    last = 0
    for m in CODE_BLOCK_PATTERN.finditer(full_text):
        lang = (m.group(1) or "").strip()
        body = m.group(2) or ""
        st, ed = m.span()
        if st > last:
            txt = full_text[last:st].strip()
            if txt:
                res.append(("text", txt, ""))
        res.append(("code", body.strip(), lang))
        last = ed
    if last < len(full_text):
        tail = full_text[last:].strip()
        if tail:
            res.append(("text", tail, ""))
    return res

def process_code_block(code_body: str, lang_hint: str, base_dir: str, log_widget):
    now_ = datetime.datetime.now().strftime('%H%M%S')
    if lang_hint.lower() == "python":
        outf = os.path.join(base_dir, f"pasted_python_{now_}.py")
        write_file_content(outf, code_body, log_widget)
        log_info(log_widget, f"Pythonコード保存:{outf}")
    elif lang_hint.lower() == "json":
        try:
            obj = json.loads(code_body)
            ok = try_merge_into_collected(obj, base_dir, log_widget)
            if ok:
                log_info(log_widget, "JSONコードマージ完了")
            else:
                outf = os.path.join(base_dir, f"pasted_json_{now_}.json")
                write_file_content(outf, code_body, log_widget)
                log_info(log_widget, f"JSON(マージ失敗)保存:{outf}")
        except:
            outf = os.path.join(base_dir, f"pasted_json_{now_}.json")
            write_file_content(outf, code_body, log_widget)
            log_info(log_widget, f"JSON(パース失敗)保存:{outf}")
    else:
        outf = os.path.join(base_dir, f"pasted_{lang_hint or 'code'}_{now_}.txt")
        write_file_content(outf, code_body, log_widget)
        log_info(log_widget, f"その他コード保存:{outf}")

def process_text_block(txt_body: str, base_dir: str, log_widget):
    now_ = datetime.datetime.now().strftime('%H%M%S')
    try:
        obj = json.loads(txt_body)
        ok = try_merge_into_collected(obj, base_dir, log_widget)
        if ok:
            log_info(log_widget, "テキスト->JSONマージ成功")
            return
    except:
        pass
    outf = os.path.join(base_dir, f"pasted_text_{now_}.txt")
    write_file_content(outf, txt_body, log_widget)
    snippet = txt_body[:50].replace("\n", " ")
    log_info(log_widget, f"テキスト保存:{outf} (先頭50文字:{snippet}...)")

def try_merge_into_collected(new_data, base_dir: str, log_widget) -> bool:
    if isinstance(new_data, list):
        new_data = {"scripts": new_data}
    if not isinstance(new_data, dict):
        return False
    cjson = os.path.join(base_dir, "collected_scripts.json")
    if os.path.isfile(cjson):
        try:
            with open(cjson, 'r', encoding='utf-8') as fr:
                old = json.load(fr)
            if not isinstance(old, dict):
                old = {"system_overview": "", "settings": {}, "scripts": []}
        except:
            old = {"system_overview": "", "settings": {}, "scripts": []}
    else:
        old = {"system_overview": "", "settings": {}, "scripts": []}
    target_ext = ('.py',)
    exclude_dirs = {'venv', '__pycache__', '.git', 'temp_build', 'backups', DATA_FOLDER_NAME}
    curr_script = os.path.abspath(__file__)
    file_map = {}
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if f in {"aum.py", "collected_scripts.json"}:
                continue
            if any(f.endswith(e) for e in target_ext):
                fp = os.path.join(root, f)
                if fp == curr_script:
                    continue
                txt = read_file_content(fp, log_widget)
                if txt:
                    rp = os.path.relpath(fp, base_dir)
                    ovv = generate_summary(fp, txt)
                    file_map[rp] = {"content": txt, "overview": ovv}
    merged = merge_collected_scripts(old, new_data, file_map, log_widget)
    try:
        with open(cjson, 'w', encoding='utf-8') as fw:
            json.dump(merged, fw, ensure_ascii=False, indent=2)
        log_info(log_widget, "collected_scripts.jsonを更新しました")
        return True
    except Exception as e:
        log_error(log_widget, f"JSONマージ失敗: {e}")
        return False

###############################################################################
# GUIエントリ
###############################################################################
def gui_main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_state = AUMAppState()
    root = tk.Tk()
    root.title("AUM GUI with Ordered Build")
    frm_main = tk.Frame(root)
    frm_main.pack(fill=tk.BOTH, expand=True)

    log_widget = ScrolledText(frm_main, width=90, height=25)
    log_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    frm_top = tk.Frame(frm_main)
    frm_top.pack(side=tk.TOP, fill=tk.X)

    def on_collect():
        collect(base_dir, log_widget)
    def on_token_saving():
        token_saving_build_unified(base_dir, log_widget, app_state)
    def on_restore():
        restore_build(base_dir, log_widget, app_state)
    def on_clean():
        clean(base_dir, log_widget)
    def on_run():
        run_main_py(base_dir, log_widget, app_state)
    def on_paste():
        enhanced_smart_paste(base_dir, log_widget)
    def on_svn():
        on_svn_commit(log_widget)

    tk.Button(frm_top, text="コード収集", command=on_collect).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(frm_top, text="トークン節約(衝突検知)", command=on_token_saving).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(frm_top, text="復元ビルド", command=on_restore).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(frm_top, text="クリーンアップ", command=on_clean).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(frm_top, text="main.py実行", command=on_run).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Button(frm_top, text="Enhanced Smart Paste", command=on_paste).pack(side=tk.LEFT, padx=10, pady=5)
    tk.Button(frm_top, text="SVNコミット", command=on_svn).pack(side=tk.LEFT, padx=5, pady=5)

    patch_ui(frm_main, base_dir, log_widget)
    root.mainloop()

import sys
import multiprocessing

def set_multiprocessing_start_method():
    try:
        if sys.platform.startswith("win"):
            multiprocessing.set_start_method("spawn", force=True)
        else:
            multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        # 既に start_method が設定されている場合は何もしない
        pass

if __name__ == "__main__":
    set_multiprocessing_start_method()
    # ここに main 処理（例: gui_main()）を呼び出す
    gui_main()
