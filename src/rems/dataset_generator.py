import json
import random
import urllib.request
import os
import re

def download_and_preprocess():
    local_path = os.path.join(os.getcwd(), "data", "hongloumeng_raw.txt")
    
    # Try to read local first
    if os.path.exists(local_path):
        print(f"Reading from local cache: {local_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        url = "https://www.gutenberg.org/cache/epub/24264/pg24264.txt"
        print(f"Downloading {url}...")
        try:
            with urllib.request.urlopen(url) as response:
                content = response.read().decode('utf-8-sig')
            # Cache it
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            print(f"Failed to download: {e}")
            return None

    # Strip Gutenberg header and footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK 紅樓夢 ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK 紅樓夢 ***"
    
    start_idx = content.find(start_marker)
    if start_idx != -1:
        content = content[start_idx + len(start_marker):]
    
    end_idx = content.find(end_marker)
    if end_idx != -1:
        content = content[:end_idx]
        
    # Clean up
    content = content.replace('\r\n', '\n').strip()
    return content

def chunk_text(text, min_size=300, max_size=2000):
    chunks = []
    total_len = len(text)
    current_pos = 0
    
    print(f"Total characters: {total_len}")
    
    while current_pos < total_len:
        if total_len - current_pos <= max_size:
            chunk_size = total_len - current_pos
        else:
            target_size = random.randint(min_size, max_size)
            split_idx = current_pos + target_size
            
            # Find nearest period '。'
            prev_period = text.rfind('。', current_pos, split_idx + 1)
            next_period = text.find('。', split_idx)
            
            if prev_period != -1 and prev_period > current_pos + min_size // 2:
                if next_period != -1:
                    if (split_idx - prev_period) <= (next_period - split_idx):
                        chunk_size = prev_period - current_pos + 1
                    else:
                        chunk_size = next_period - current_pos + 1
                else:
                    chunk_size = prev_period - current_pos + 1
            elif next_period != -1:
                chunk_size = next_period - current_pos + 1
            else:
                chunk_size = target_size
        
        # Check for trailing quotes or other punctuation
        while current_pos + chunk_size < total_len and text[current_pos + chunk_size] in '”』〉"\'':
            chunk_size += 1
            
        chunk = text[current_pos : current_pos + chunk_size]
        chunks.append({
            "id": len(chunks) + 1,
            "size": len(chunk),
            "content": chunk
        })
        current_pos += chunk_size
        # Limit print output to avoid huge logs
        if len(chunks) % 50 == 0:
            print(f"Created {len(chunks)} chunks...")
        
    return chunks

def main():
    text = download_and_preprocess()
    if not text:
        # Fallback if download failed but we have some previous content or we can try a different approach
        print("Error: Could not obtain text.")
        return
        
    chunks = chunk_text(text)
    
    data_dir = os.path.join(os.getcwd(), "data")
    output_path = os.path.join(data_dir, "hongloumeng_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print(f"Dataset generated with {len(chunks)} chunks and saved to {output_path}")

if __name__ == "__main__":
    main()
