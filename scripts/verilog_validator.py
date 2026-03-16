import pandas as pd
import subprocess
import os
import tempfile
import concurrent.futures
from tqdm import tqdm

VERILOG_CMD = r"C:\iverilog\bin\iverilog.exe"

def check_syntax(code_str, idx):
    if not isinstance(code_str, str) or not code_str.strip():
        return False, 0
    
    # Calculate complexity (lines of code)
    lines = len(code_str.split('\n'))
    
    # Write to temp file and test
    temp_filename = f"temp_{idx}.v"
    try:
        with open(temp_filename, "w", encoding="utf-8") as f:
            f.write(code_str)
        
        # Run iverilog syntax check (null target means it just checks syntax/semantics)
        result = subprocess.run([VERILOG_CMD, "-t", "null", temp_filename], 
                                capture_output=True, text=True, timeout=5)
        passed = (result.returncode == 0)
    except Exception as e:
        passed = False
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
    return passed, lines

def get_complexity_category(lines):
    if lines < 50:
        return 'Low'
    elif lines <= 200:
        return 'Medium'
    else:
        return 'High'

def main():
    print("Loading CSV...")
    df = pd.read_csv("christon_internet_set_cleaned.csv")
    
    # We want 1000 of each if possible, total 3000
    TARGET_PER_CATEGORY = 1000
    
    low_passed = []
    med_passed = []
    high_passed = []
    
    print(f"Total entries to process: {len(df)}")
    
    # Use ThreadPoolExecutor to speed up compilation tests
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(check_syntax, row['code'], i): (i, row) for i, row in df.iterrows()}
        
        pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Verifying Syntax")
        for future in pbar:
            i, row = futures[future]
            try:
                passed, lines = future.result()
                if passed:
                    cat = get_complexity_category(lines)
                    row_with_cat = row.copy()
                    row_with_cat['complexity_lines'] = lines
                    row_with_cat['complexity_category'] = cat
                    
                    if cat == 'Low' and len(low_passed) < TARGET_PER_CATEGORY:
                        low_passed.append(row_with_cat)
                    elif cat == 'Medium' and len(med_passed) < TARGET_PER_CATEGORY:
                        med_passed.append(row_with_cat)
                    elif cat == 'High' and len(high_passed) < TARGET_PER_CATEGORY:
                        high_passed.append(row_with_cat)
                        
                    pbar.set_postfix({"L": len(low_passed), "M": len(med_passed), "H": len(high_passed)})
                    
                    # Stop if we hit 1000 for each (3000 total)
                    if len(low_passed) >= TARGET_PER_CATEGORY and len(med_passed) >= TARGET_PER_CATEGORY and len(high_passed) >= TARGET_PER_CATEGORY:
                        print("Reached 3000 total valid files across all complexity levels!")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
            except Exception as e:
                pass
                
    result_df = pd.DataFrame(low_passed + med_passed + high_passed)
    print(f"\nFinal dataset distribution:")
    print(result_df['complexity_category'].value_counts())
    
    output_file = "filtered_verilog_dataset_3000.csv"
    result_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file} (Total: {len(result_df)})")

if __name__ == "__main__":
    main()
