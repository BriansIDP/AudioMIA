import json
import yt_dlp

def search_with_ytdlp(json_file_path, output_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # yt-dlp options: extract metadata only, filter by date, suppress output
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        "extract_flat": True,      # Don't download, just get metadata
        'dateafter': '20260101',   # STRICT 2026+ FILTER (YYYYMMDD format)
    }

    all_urls = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for category in data['categories']:
            print(f"\n--- Processing Category: {category['category_name']} ---")
            
            for q in category['queries']:
                query_text = q['query_text'] + " after:2026-01-01"
                print(f"Searching: {query_text}...")
                
                # Search for 15 results to ensure we get at least 10 after date filtering
                search_query = f"ytsearch100:{query_text}" 
                
                try:
                    info = ydl.extract_info(search_query, download=False)
                    entries = info.get('entries', [])
                    
                    for entry in entries:
                        # Double-check upload date just in case
                        upload_date = entry.get('upload_date')
                        # if upload_date and upload_date >= '20260101':
                        video_id = entry.get('id')
                        url = f"https://www.youtube.com/watch?v={video_id}"
                        
                        all_urls.append({
                            'category_id': category['category_id'],
                            'category_name': category['category_name'],
                            'query': query_text,
                            'url': url,
                            # 'upload_date': upload_date
                        })
                except Exception as e:
                    print(f"  [Error] Failed to search '{query_text}': {e}")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_urls, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Success! Saved {len(all_urls)} URLs to {output_file_path}")

if __name__ == "__main__":
    search_with_ytdlp('search_queries.json', 'youtube_urls_ytdlp.json')