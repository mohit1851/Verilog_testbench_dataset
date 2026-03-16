import os
import subprocess
import shutil
import pandas as pd
from tqdm import tqdm

# Configuration
RESULTS_DIR = "./benchmark_results"
GEMINI_RESULTS_FILE = "gemini_final_combined.csv"
EVAL_SUMMARY_FILE = "final_benchmark_summary.csv"
FINAL_REPORT_FILE = "final_summary.md"

def find_iverilog():
    """Auto-detect iverilog path (Cross-platform)."""
    # 1. Common Windows Install Paths
    for path in [
        r"C:\iverilog\bin\iverilog.exe",
        r"C:\iverilog\iverilog.exe"
    ]:
        if os.path.exists(path):
            return path

    # 2. Check system PATH (standard for Cloud/Linux)
    iverilog = shutil.which("iverilog")
    if iverilog:
        return iverilog
        
    # 3. Linux/Unix Fallbacks
    for path in ["/usr/bin/iverilog", "/usr/local/bin/iverilog"]:
        if os.path.exists(path):
            return path
            
    return None

def compute_gemini_baseline():
    """Dynamically compute Gemini baseline from the actual combined results file."""
    if not os.path.exists(GEMINI_RESULTS_FILE):
        print(f"WARNING: {GEMINI_RESULTS_FILE} not found. Gemini baseline will be skipped.")
        return None
    
    print(f"Loading Gemini results from {GEMINI_RESULTS_FILE}...")
    df_full = pd.read_csv(GEMINI_RESULTS_FILE)
    
    # Filter to only include designs that were eventually passed (the 'gold' subset)
    # This ensures apples-to-apples with FT models trained on this rescued data.
    df = df_full[df_full['passed_verification'] == True].copy()
    
    total = len(df)
    zero_shot = len(df[df['retries_needed'] == 0])
    self_healed = len(df[df['retries_needed'] > 0])
    
    print(f"  Gemini Baseline Stats (Filtered to Gold Subset):")
    print(f"    Total successful designs: {total}")
    print(f"    - Zero-shot passes:       {zero_shot} ({zero_shot/total*100:.2f}%)")
    print(f"    - Self-healed passes:     {self_healed} ({self_healed/total*100:.2f}%)")
    
    return {
        "total": total,
        "passed": total,
        "zero_shot": zero_shot,
        "self_healed": self_healed,
        "failed": 0,
        "pass_rate": 100.0,
        "zero_shot_rate": round((zero_shot / total) * 100, 2) if total > 0 else 0
    }

import re

def is_gibberish(text):
    """Checks if the text contains BPE artifacts or excessive non-ASCII chars."""
    if not isinstance(text, str): return True
    # Count non-ASCII characters in first 200 chars
    sample = text[:200]
    non_ascii = len([c for c in sample if ord(c) > 127])
    return non_ascii > (len(sample) * 0.1) # More than 10% non-ascii is likely gibberish

def verify_testbench_custom(iverilog_path, design_code, testbench_code, temp_file):
    """Verifies Verilog code using iverilog, with robust code extraction."""
    
    if not isinstance(testbench_code, str) or testbench_code.startswith("ERROR"):
        return False
        
    # Robust extraction: look for ```verilog ... ``` blocks first
    code_match = re.search(r"```verilog\s+(.*?)\s+```", testbench_code, re.DOTALL | re.IGNORECASE)
    if not code_match:
        # Fallback to general code blocks
        code_match = re.search(r"```\s+(.*?)\s+```", testbench_code, re.DOTALL)
    
    clean_tb = code_match.group(1) if code_match else testbench_code
    
    # Strip any remaining "Here is your code" style headers if they leaked
    if "module" in clean_tb:
        module_start = clean_tb.find("module")
        # Find the last endmodule
        module_end = clean_tb.rfind("endmodule")
        if module_end != -1:
            clean_tb = clean_tb[module_start:module_end + 9]
    
    combined = str(design_code) + "\n\n" + str(clean_tb)
    
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(combined)
        
        result = subprocess.run(
            [iverilog_path, "-t", "null", temp_file],
            capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        # Retry removal to handle Windows PermissionError (file lock)
        import time
        for _ in range(5):
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                break
            except PermissionError:
                time.sleep(0.5)

def main():
    iverilog_path = find_iverilog()
    if not iverilog_path:
        print("ERROR: iverilog not found. Install it or add to PATH.")
        print("  Ubuntu: sudo apt-get install iverilog")
        print("  Windows: Download from http://bleyer.org/icarus/")
        return

    print(f"Using iverilog at: {iverilog_path}")

    # Step 1: Compute Gemini baseline from actual results
    gemini_stats = compute_gemini_baseline()

    # Step 2: Evaluate fine-tuned models
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: {RESULTS_DIR} not found.")
        return

    all_stats = []
    
    for filename in sorted(os.listdir(RESULTS_DIR)):
        if filename.startswith("results_") and filename.endswith(".csv"):
            model_id = filename.replace("results_", "").replace(".csv", "")
            
            # Determine Variant
            if "base_corners" in model_id:
                variant = "Variant 2: Base + Corners"
            elif "base" in model_id:
                variant = "Variant 1: Base Model"
            else:
                variant = "Variant 3: LoRA Fine-Tuned"
                
            print(f"\nEvaluating metrics for {model_id} ({variant})...")
            
            df = pd.read_csv(os.path.join(RESULTS_DIR, filename))
            passed_count = 0
            error_count = 0
            
            # Use model-specific prefix for temp files to avoid collisions if any
            model_prefix = model_id.replace(".", "_")
            
            # Check for corruption but don't skip (report 0% if it fails)
            if is_gibberish(df['generated_testbench'].dropna().iloc[0]):
                print(f"  WARNING: {model_id} data appears corrupted (gibberish detected). Result will likely be 0%.")

            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Validating {model_id}"):
                tb = row.get('generated_testbench', '')
                if isinstance(tb, str) and tb.startswith("ERROR"):
                    continue
                
                temp_name = f"temp_{model_prefix}_{idx}.v"
                if verify_testbench_custom(iverilog_path, row['code'], tb, temp_name):
                    passed_count += 1
            
            pass_rate = (passed_count / len(df)) * 100 if len(df) > 0 else 0
            print(f"Result for {model_id}: {passed_count}/{len(df)} passed ({pass_rate:.2f}%)")
            
            all_stats.append({
                "Variant": variant,
                "Model": model_id.replace("base_corners_", "").replace("base_", ""),
                "Total": len(df),
                "Passed": passed_count,
                "Pass_Rate": round(pass_rate, 2)
            })

    # Add Gemini baseline row(s)
    if gemini_stats:
        all_stats.append({
            "Variant": "Industry Baseline",
            "Model": "Gemini 2.5 (Zero-Shot)",
            "Total": gemini_stats["total"],
            "Passed": gemini_stats["zero_shot"],
            "Pass_Rate": gemini_stats["zero_shot_rate"]
        })
        all_stats.append({
            "Variant": "Industry Baseline",
            "Model": "Gemini 2.5 (Self-Healed)",
            "Total": gemini_stats["total"],
            "Passed": gemini_stats["passed"],
            "Pass_Rate": gemini_stats["pass_rate"]
        })

    # Save CSV summary
    summary_df = pd.DataFrame(all_stats)
    summary_df = summary_df.sort_values(["Variant", "Pass_Rate"], ascending=[True, False])
    summary_df.to_csv(EVAL_SUMMARY_FILE, index=False)
    print(f"\nEvaluation summary saved to {EVAL_SUMMARY_FILE}")
    
    # Generate Markdown report
    generate_report(summary_df, gemini_stats)

def generate_report(summary_df, gemini_stats):
    """Generate a human-readable markdown report."""
    report = "# Final Benchmark Summary: LoRA Fine-Tuned Models vs Gemini Baseline\n\n"
    report += "## Test Configuration\n"
    report += f"- **Fine-tuned models tested on**: 100 held-out Verilog designs\n"
    if gemini_stats:
        report += f"- **Gemini baseline evaluated on**: {gemini_stats['total']} designs from `gemini_final_combined.csv`\n"
        report += f"- **Gemini zero-shot pass rate**: {gemini_stats['zero_shot_rate']}% ({gemini_stats['zero_shot']}/{gemini_stats['total']})\n"
        report += f"- **Gemini self-healed pass rate**: {gemini_stats['pass_rate']}% ({gemini_stats['passed']}/{gemini_stats['total']})\n"
    report += f"- **Evaluation Method**: `iverilog` compilation (syntax + structural correctness)\n\n"
    
    report += "## Results\n\n"
    report += "| Variant | Model | Total | Passed | Pass Rate (%) |\n"
    report += "|---------|-------|-------|--------|---------------|\n"
    
    for _, row in summary_df.iterrows():
        report += f"| {row['Variant']} | {row['Model']} | {row['Total']} | {row['Passed']} | {row['Pass_Rate']}% |\n"
    
    report += "\n## Key Findings\n\n"
    
    ft_models = summary_df[~summary_df['Model'].str.contains('gemini')]
    if len(ft_models) > 0:
        best = ft_models.loc[ft_models['Pass_Rate'].idxmax()]
        report += f"- **Best Fine-Tuned Model**: {best['Model']} with {best['Pass_Rate']}% Pass@1\n"
        
        if gemini_stats:
            diff = best['Pass_Rate'] - gemini_stats['zero_shot_rate']
            if diff > 0:
                report += f"- **Improvement over Gemini zero-shot**: +{diff:.2f} percentage points\n"
            else:
                report += f"- **Gap vs Gemini zero-shot**: {diff:.2f} percentage points\n"
    
    report += "\n---\n*Generated automatically by evaluate_benchmarks.py*\n"
    
    with open(FINAL_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Final report saved to {FINAL_REPORT_FILE}")

if __name__ == "__main__":
    main()
