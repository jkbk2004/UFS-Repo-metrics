import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from time import sleep

# Repository to analyze
REPO = "ufs-community/ufs-weather-model"

# Optional GitHub token for higher rate limits
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

def fetch_prs(state="closed", per_page=100, max_pages=10):
    """
    Fetches pull requests from GitHub API.

    Args:
        state (str): 'open', 'closed', or 'all'
        per_page (int): PRs per page
        max_pages (int): Max number of pages to fetch

    Returns:
        list of dicts: PR metadata
    """
    prs = []
    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/repos/{REPO}/pulls"
        params = {"state": state, "per_page": per_page, "page": page}
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break
        data = response.json()
        if not data:
            break
        prs.extend(data)
        sleep(0.5)  # Rate limit buffer
    return prs

def compute_turnaround(prs):
    """
    Computes turnaround time for each PR.

    Args:
        prs (list): List of PR metadata

    Returns:
        pd.DataFrame: PRs with turnaround time in hours
    """
    records = []
    for pr in prs:
        created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
        closed = pr.get("closed_at")
        merged = pr.get("merged_at")
        if closed:
            closed_dt = datetime.fromisoformat(closed.replace("Z", "+00:00"))
            turnaround = (closed_dt - created).total_seconds() / 3600
            records.append({
                "number": pr["number"],
                "title": pr["title"],
                "created_at": created,
                "closed_at": closed_dt,
                "merged": bool(merged),
                "turnaround_hours": round(turnaround, 2)
            })
    return pd.DataFrame(records)

def summarize(df):
    """
    Prints summary statistics for merged PRs.

    Args:
        df (pd.DataFrame): PR turnaround data
    """
    merged = df[df["merged"]]
    print(f"🔁 Total merged PRs: {len(merged)}")
    print(f"⏱ Average turnaround: {merged['turnaround_hours'].mean():.2f} hours")
    print(f"📊 Median turnaround: {merged['turnaround_hours'].median():.2f} hours")
    print(f"🚀 Fastest turnaround: {merged['turnaround_hours'].min():.2f} hours")
    print(f"🐢 Slowest turnaround: {merged['turnaround_hours'].max():.2f} hours")

def plot_turnaround(df, output_path="ufs_pr_turnaround_plot.png"):
    """
    Plots PR number vs. turnaround time in days.

    Args:
        df (pd.DataFrame): PR turnaround data
        output_path (str): Path to save the plot
    """
    df["turnaround_days"] = df["turnaround_hours"] / 24
    merged = df[df["merged"]]

    plt.figure(figsize=(10, 6))
    plt.scatter(merged["number"], merged["turnaround_days"], color="blue", alpha=0.7)
    plt.title("PR Turnaround Time (Days) – ufs-weather-model")
    plt.xlabel("PR Number")
    plt.ylabel("Turnaround Time (Days)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    print("🔍 Fetching PR data...")
    prs = fetch_prs()
    print(f"✅ Retrieved {len(prs)} pull requests")
    df = compute_turnaround(prs)
    summarize(df)
    df.to_csv("ufs_pr_turnaround.csv", index=False)
    print("📁 Saved results to ufs_pr_turnaround.csv")
    plot_turnaround(df)
    print("📈 Saved plot to ufs_pr_turnaround_plot.png")
