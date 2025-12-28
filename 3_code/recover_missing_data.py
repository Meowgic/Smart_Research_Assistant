import os
import csv
from pathlib import Path
import arxiv
from zoneinfo import ZoneInfo
from datetime import datetime

# 配置路径（和主程序一致）
PDF_DIR = Path("../1_raw_documents/pdf_files")
META_FILE = Path("../1_raw_documents/metadata/metadata.csv")
FAILED_FILE = Path("../1_raw_documents/metadata/failed_papers.csv")

# 步骤1：获取已下载的PDF ID列表
pdf_ids = set()
for file in PDF_DIR.glob("*.pdf"):
    if file.suffix == ".pdf":
        paper_id = file.stem  # 提取文件名（去掉.pdf）
        pdf_ids.add(paper_id)
print(f"已下载PDF数量：{len(pdf_ids)}")

# 步骤2：获取已记录在CSV中的ID列表
csv_ids = set()
if META_FILE.exists():
    with open(META_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('id'):
                csv_ids.add(row['id'])
print(f"已记录在CSV的ID数量：{len(csv_ids)}")

# 步骤3：找出缺失的ID（下载了但未记录）
missing_ids = pdf_ids - csv_ids
print(f"缺失记录的ID数量：{len(missing_ids)}")

# 步骤4：批量补录缺失的元数据
if missing_ids:
    # 初始化arxiv客户端（低频率，避免限流）
    client = arxiv.Client(
        page_size=50,
        delay_seconds=5,
        num_retries=3
    )
    # 准备补录的元数据
    missing_meta = []
    fieldnames = ['id', 'title', 'authors', 'abstract', 'categories', 'submit_date', 'pdf_path']
    
    for i, paper_id in enumerate(missing_ids):
        print(f"补录 {i+1}/{len(missing_ids)}: {paper_id}")
        try:
            # 根据ID查询论文元数据
            search = arxiv.Search(id_list=[paper_id])
            paper = next(client.results(search))
            
            # 构造和主程序一致的元数据
            meta = {
                'id': paper_id,
                'title': paper.title,
                'authors': ', '.join([a.name for a in paper.authors]),
                'abstract': paper.summary.strip(),
                'categories': ', '.join(paper.categories),
                'submit_date': paper.published.isoformat(),
                'pdf_path': str(PDF_DIR / f"{paper_id}.pdf")
            }
            missing_meta.append(meta)
        except Exception as e:
            # 补录失败则记录到failed_file
            print(f"补录失败 {paper_id}: {e}")
            with open(FAILED_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'error', 'timestamp'])
                if not FAILED_FILE.exists():
                    writer.writeheader()
                writer.writerow({
                    'id': paper_id,
                    'error': f"补录元数据失败: {str(e)}",
                    'timestamp': datetime.now(ZoneInfo("UTC")).isoformat()
                })
    
    # 批量写入缺失的元数据到CSV
    if missing_meta:
        with open(META_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # 若CSV为空则先写表头
            if not META_FILE.exists():
                writer.writeheader()
            writer.writerows(missing_meta)
        print(f"成功补录 {len(missing_meta)} 条元数据")
else:
    print("无缺失的元数据记录")