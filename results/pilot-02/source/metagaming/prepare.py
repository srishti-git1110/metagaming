"""download the small pinned prompt files and rubric; no inference"""
from .data import DATA_DIR, prepare

if __name__ == "__main__":
    prepare()
    print(f"Verified 444 prompts and the original VEA rubric in {DATA_DIR}")
