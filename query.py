"""
query.py — Query the School Intelligence Database (hybrid dense+sparse)

Usage:
    python query.py "vendor contracts expiring soon"
    python query.py "roof problems" --type problem
    python query.py --school "Auckland Academy" --type vendor
    python query.py --list-schools
"""

import argparse
from loguru import logger

from embedder import get_embedder, SparseVectorizer
from uploader import QdrantUploader


def search(query: str, entity_type: str = None, school_name: str = None, limit: int = 10):
    embedder = get_embedder()
    db       = QdrantUploader()

    [query_vector] = embedder.embed([query])
    sparse_indices, sparse_values = SparseVectorizer.vectorize(query)

    filters = {}
    if entity_type:  filters["type"]        = entity_type
    if school_name:  filters["school_name"] = school_name

    results = db.search(
        query_vector=query_vector,
        filter_conditions=filters or None,
        limit=limit,
        query_sparse=(sparse_indices, sparse_values),
    )

    print(f"\n{'='*60}")
    print(f"Query: '{query}'" + (f" | Filters: {filters}" if filters else ""))
    print(f"Results: {len(results)}\n{'='*60}")

    for i, r in enumerate(results, 1):
        score = r.pop("_score", 0)
        etype = r.get("type", "")
        print(f"\n[{i}] Score: {score:.3f} | {etype.upper()} | {r.get('school_name')}")
        print(f"    {r.get('source_label', '')}")

        if etype == "vendor":
            print(f"    Vendor: {r.get('vendor_name')} | {r.get('service_type')}")
            print(f"    Value: {r.get('contract_value')} | Expiry: {r.get('expiry_date')}")
        elif etype == "budget":
            print(f"    {r.get('category')} | {r.get('currency')} {r.get('amount')}")
            print(f"    Source: {r.get('funding_source')} | Status: {r.get('status')}")
        elif etype == "project":
            print(f"    {r.get('project_name')} | {r.get('value')} | {r.get('status')}")
        elif etype == "problem":
            print(f"    [{r.get('severity')}] {r.get('category')}: {r.get('raw_text','')[:100]}")
        elif etype == "board_member":
            print(f"    {r.get('name')} | {r.get('role')} | {r.get('term_start')}→{r.get('term_end')}")
        elif etype == "contractor":
            print(f"    {r.get('contractor_name')} | {r.get('trade')} | Expiry: {r.get('expiry_date')}")


def list_schools():
    db      = QdrantUploader()
    results = db.filter_only({}, limit=1000)
    schools: dict[str, dict] = {}
    for r in results:
        name  = r.get("school_name", "unknown")
        etype = r.get("type", "unknown")
        schools.setdefault(name, {})
        schools[name][etype] = schools[name].get(etype, 0) + 1

    print(f"\nSchools in database: {len(schools)}\n{'='*60}")
    for school, counts in sorted(schools.items()):
        total = sum(counts.values())
        breakdown = " | ".join(f"{t}: {n}" for t, n in counts.items())
        print(f"\n  {school} ({total} records)\n    {breakdown}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the School Intelligence Database")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--type",         help="Filter: vendor|budget|project|problem|board_member|contractor")
    parser.add_argument("--school",       help="Filter by school name")
    parser.add_argument("--limit",        type=int, default=10)
    parser.add_argument("--list-schools", action="store_true")
    args = parser.parse_args()

    if args.list_schools:
        list_schools()
    elif args.query:
        search(args.query, entity_type=args.type, school_name=args.school, limit=args.limit)
    else:
        parser.print_help()
