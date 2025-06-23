import os
import json

def aggregate_triplets_json(folder_path, output_path="final_triplets_aggregated.json"):
    all_triplets = []

    for filename in os.listdir(folder_path):
        if filename.endswith((".json", ".JSON")):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "triplets" in data and isinstance(data["triplets"], list):
                        all_triplets.extend(data["triplets"])
                        print(f"✅ Loaded: {filename} ({len(data['triplets'])} triplets)")
                    else:
                        print(f"⚠️ File ignored (no 'triplets' key): {filename}")
            except Exception as e:
                print(f"❌ Read error in {filename}: {e}")

    final_data = {"triplets": all_triplets}

    with open(os.path.join(folder_path, output_path), "w", encoding="utf-8") as f_out:
        json.dump(final_data, f_out, indent=2, ensure_ascii=False)

    print(f"\n✅ All JSON files aggregated into: {output_path}")
    print(f"📊 Total triplets: {len(all_triplets)}")


# aggregate_triplets_json(output_dir)
