import argparse
import logging
import time
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import arxiv
import csv
from pathlib import Path

# Basic logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('download_log.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Core config (all key parameters retained + categories config)
CFG = {
    "pdf_dir": Path("../1_raw_documents/pdf_files"),
    "meta_dir": Path("../1_raw_documents/metadata"),
    "meta_file": Path("../1_raw_documents/metadata/metadata.csv"),
    "failed_file": Path("../1_raw_documents/metadata/failed_papers.csv"),
    "max_retries": 3,
    "req_interval": 2,
    "timeout": 60,
    "batch_size": 100,
    "target_total": 120000,
    "categories": "(cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.NE)"  # Configurable categories
}

# Create dirs & set global timeout
for dir_path in [CFG["pdf_dir"], CFG["meta_dir"]]:
    dir_path.mkdir(parents=True, exist_ok=True)
requests.packages.urllib3.util.timeout.Timeout.DEFAULT_TIMEOUT = CFG["timeout"]

# ------------------------------
# Simplified helper functions
# ------------------------------
def load_ids(file_path, is_set=True):
    """Load paper IDs from CSV (set for downloaded, list for failed)"""
    result = set() if is_set else []
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('id'):
                        if is_set:
                            result.add(row['id'])
                        else:
                            result.append(row['id'])
            logger.info(f"Loaded {len(result)} IDs from {file_path.name}")
        except Exception as e:
            logger.error(f"Load error {file_path}: {str(e)}")
    return result

def save_csv(data, file_path, fieldnames):
    """Save data to CSV (append mode with header handling)"""
    if not data:
        return
    file_exists = file_path.exists()
    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(data)
        logger.info(f"Saved {len(data)} entries to {file_path.name}")
    except Exception as e:
        logger.error(f"Save error {file_path}: {str(e)}")

def download_single(paper, downloaded_ids):
    """Download single paper (removed download_time)"""
    paper_id = paper.get_short_id()
    if paper_id in downloaded_ids:
        return None, False

    # Metadata structure without download_time
    meta = {
        'id': paper_id,
        'title': paper.title,
        'authors': ', '.join([a.name for a in paper.authors]),
        'abstract': paper.summary.strip(),
        'categories': ', '.join(paper.categories),
        'submit_date': paper.published.isoformat(),
        'pdf_path': str(CFG["pdf_dir"] / f"{paper_id}.pdf")
    }

    # Download process
    try:
        logger.info(f"Downloading {paper_id}")
        paper.download_pdf(filename=meta['pdf_path'])
        return meta, True
    except Exception as e:
        # 失败记录中仍保留时间戳（用于调试），但元数据中已移除
        save_csv([{'id': paper_id, 'error': str(e), 'timestamp': datetime.now(ZoneInfo("UTC")).isoformat()}],
                 CFG["failed_file"], ['id', 'error', 'timestamp'])
        logger.error(f"Failed {paper_id}: {str(e)}")
        return None, False

# ------------------------------
# Core window calculation & download
# ------------------------------
def calc_next_window(current_start, current_end, global_start, window_days):
    """Calculate next earlier time window"""
    next_end = current_start - timedelta(days=1)
    next_start = next_end - timedelta(days=window_days)
    if next_start < global_start:
        next_start = global_start
        return (None, None) if next_start > next_end else (next_start, next_end)
    return next_start, next_end

def download_window(window_start, window_end, downloaded_ids, total_downloaded):
    """Download papers in time window (removed download_time from CSV)"""
    start_str = window_start.astimezone(ZoneInfo("UTC")).strftime('%Y%m%d')
    end_str = window_end.astimezone(ZoneInfo("UTC")).strftime('%Y%m%d')
    query = f"{CFG['categories']} AND submittedDate:[{start_str} TO {end_str}]"
    
    logger.info(f"===== Downloading window: {window_start} to {window_end} =====")
    client = arxiv.Client(
        page_size=CFG["batch_size"], 
        delay_seconds=CFG["req_interval"], 
        num_retries=CFG["max_retries"]
    )
    search = arxiv.Search(
        query=query, 
        max_results=float('inf'), 
        sort_by=arxiv.SortCriterion.SubmittedDate, 
        sort_order=arxiv.SortOrder.Descending
    )
    
    meta_batch = []
    batch_count = 0

    try:
        for paper in client.results(search):
            if total_downloaded >= CFG["target_total"]:
                logger.info(f"Reached target {CFG['target_total']} - stopping")
                break

            meta, success = download_single(paper, downloaded_ids)
            if success:
                meta_batch.append(meta)
                total_downloaded += 1
                downloaded_ids.add(paper.get_short_id())

            # Batch save (removed download_time from fieldnames)
            batch_count += 1
            if batch_count % CFG["batch_size"] == 0:
                save_csv(
                    meta_batch, 
                    CFG["meta_file"], 
                    ['id', 'title', 'authors', 'abstract', 'categories', 'submit_date', 'pdf_path']
                )
                meta_batch = []
                logger.info(f"Batch done - New: {batch_count}, Total: {total_downloaded}")
            time.sleep(CFG["req_interval"])

        # Save remaining metadata (removed download_time)
        save_csv(
            meta_batch, 
            CFG["meta_file"], 
            ['id', 'title', 'authors', 'abstract', 'categories', 'submit_date', 'pdf_path']
        )
    except Exception as e:
        logger.error(f"Window error: {str(e)}")
        # Retry logic
        for retry in range(CFG["max_retries"]):
            logger.info(f"Retry window (attempt {retry+1}/{CFG['max_retries']})")
            time.sleep(5 * (retry + 1))
            try:
                for paper in client.results(search):
                    continue
            except:
                continue

    return total_downloaded

# ------------------------------
# Retry failed papers
# ------------------------------
def retry_failed(downloaded_ids):
    """Retry failed papers (removed download_time)"""
    failed_ids = load_ids(CFG["failed_file"], is_set=False)
    if not failed_ids:
        logger.info("No failed papers to retry")
        return 0

    logger.info(f"Retrying {len(failed_ids)} failed papers...")
    client = arxiv.Client()
    retried_success = 0
    meta_batch = []

    for i, paper_id in enumerate(failed_ids):
        if paper_id in downloaded_ids:
            continue
        try:
            logger.info(f"Retry {paper_id} ({i+1}/{len(failed_ids)})")
            results = list(client.results(arxiv.Search(id_list=[paper_id])))
            if results:
                meta, success = download_single(results[0], downloaded_ids)
                if success:
                    meta_batch.append(meta)
                    retried_success += 1
                    downloaded_ids.add(paper_id)
            time.sleep(CFG["req_interval"])
        except Exception as e:
            logger.error(f"Retry failed {paper_id}: {str(e)}")

    # Save retried metadata (removed download_time)
    save_csv(
        meta_batch, 
        CFG["meta_file"], 
        ['id', 'title', 'authors', 'abstract', 'categories', 'submit_date', 'pdf_path']
    )
    logger.info(f"Retry done - Success: {retried_success}")
    return retried_success

# ------------------------------
# Main process
# ------------------------------
def main():
    """Main download process"""
    parser = argparse.ArgumentParser(description='Batch download arXiv papers')
    parser.add_argument('--global-start', type=str, default='2020-01-01', help='YYYY-MM-DD')
    parser.add_argument('--initial-end', type=str, default=None, help='YYYY-MM-DD (default: now)')
    parser.add_argument('--window-days', type=int, default=30, help='Days per window')
    parser.add_argument('--skip-retry', action='store_true', help='Skip failed retry')
    args = parser.parse_args()

    # Parse dates
    try:
        global_start = datetime.strptime(args.global_start, '%Y-%m-%d').replace(tzinfo=ZoneInfo("UTC"))
        initial_end = datetime.strptime(args.initial_end, '%Y-%m-%d').replace(tzinfo=ZoneInfo("UTC")) if args.initial_end else datetime.now(ZoneInfo("UTC"))
        window_days = args.window_days
        logger.info(f"Global start: {global_start} | Initial end: {initial_end} | Window days: {window_days} | Target: {CFG['target_total']}")
        logger.info(f"Categories filter: {CFG['categories']}")
    except ValueError as e:
        logger.error(f"Invalid date: {str(e)} (use YYYY-MM-DD)")
        return

    # Load downloaded IDs
    downloaded_ids = load_ids(CFG["meta_file"])
    total_downloaded = len(downloaded_ids)

    if total_downloaded >= CFG["target_total"]:
        logger.info(f"Already downloaded {total_downloaded} (target: {CFG['target_total']}) - exit")
        return

    # Initialize window & iterate
    current_end = initial_end
    current_start = max(current_end - timedelta(days=window_days), global_start)
    window_count = 0

    while current_start <= current_end and total_downloaded < CFG["target_total"]:
        window_count += 1
        total_downloaded = download_window(current_start, current_end, downloaded_ids, total_downloaded)
        
        # Calculate next window
        next_start, next_end = calc_next_window(current_start, current_end, global_start, window_days)
        if not next_start:
            logger.info("No more windows")
            break
        current_start, current_end = next_start, next_end
        logger.info(f"===== Switch to window {window_count + 1} =====")

    # Retry failed (if enabled)
    if not args.skip_retry and total_downloaded < CFG["target_total"]:
        total_downloaded += retry_failed(downloaded_ids)

    # Final summary
    logger.info(f"Download complete | Total: {total_downloaded}/{CFG['target_total']} | Windows processed: {window_count}")
    logger.info(f"PDFs: {CFG['pdf_dir']} | Metadata: {CFG['meta_file']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)