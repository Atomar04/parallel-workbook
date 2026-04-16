import json

input_file = 'Movies_and_TV_5.json'
output_file = 'ratings_extracted.csv'

print(f"Starting extraction from {input_file}...")

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    outfile.write("asin,overall\n")
    processed_count = 0
    for line in infile:
        try:
            record = json.loads(line.strip())
            asin = record.get('asin')
            overall = record.get('overall')
            if asin is not None and overall is not None:
                outfile.write(f"{asin},{overall}\n")
                processed_count += 1
                if processed_count % 100000 == 0:
                    print(f"Processed {processed_count} records...")
                    
        except json.JSONDecodeError:
            continue

print(f"Extraction complete! Successfully saved {processed_count} records to {output_file}.")