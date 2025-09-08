import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from time import sleep
from collections import defaultdict

REPO = "ufs-community/ufs-weather-model"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

LABEL_COLORS = {
    "bug": "#E24A33",
    "enhancement": "#348ABD",
    "documentation": "#988ED5",
    "default": "#4C72B0"
}

def fetch_prs(state="closed", per_page=100, max_pages=10):
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
        sleep(0.5)
    return prs

def compute_turnaround(prs):
    records = []
    for pr in prs:
        created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
        closed = pr.get("closed_at")
        merged = pr.get("merged_at")
        if closed:
            closed_dt = datetime.fromisoformat(closed.replace("Z", "+00:00"))
            turnaround = (closed_dt - created).total_seconds() / 3600
            labels = [label["name"] for label in pr.get("labels", [])]
            records.append({
                "number": pr["number"],
                "title": pr["title"],
                "user": pr["user"]["login"],
                "created_at": created,
                "closed_at": closed_dt,
                "merged": bool(merged),
                "labels": labels,
                "turnaround_hours": round(turnaround, 2)
            })
    return pd.DataFrame(records)

def summarize(df):
    merged = df[df["merged"]]
    print(f"🔁 Total merged PRs: {len(merged)}")
    print(f"⏱ Average turnaround: {merged['turnaround_hours'].mean():.2f} hours")
    print(f"📊 Median turnaround: {merged['turnaround_hours'].median():.2f} hours")
    print(f"🚀 Fastest turnaround: {merged['turnaround_hours'].min():.2f} hours")
    print(f"🐢 Slowest turnaround: {merged['turnaround_hours'].max():.2f} hours")

    print("\n👥 Contributor Summary:")
    by_user = merged.groupby("user")["turnaround_hours"].agg(["count", "mean", "std"]).sort_values("count", ascending=False)
    print(by_user.round(2))

def plot_turnaround(df, output_path="ufs_pr_turnaround_advanced.png", show_rolling=True):
    df["turnaround_days"] = df["turnaround_hours"] / 24
    merged = df[df["merged"]].sort_values("number")

    x = merged["number"].values
    y = merged["turnaround_days"].values

    # Assign colors based on labels
    colors = []
    for labels in merged["labels"]:
        label = next((l for l in labels if l in LABEL_COLORS), "default")
        colors.append(LABEL_COLORS[label])

    plt.figure(figsize=(14, 6))
    bars = plt.bar(x, y, color=colors, edgecolor="black")

    # Linear trend line
    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)
    plt.plot(x, trend(x), color="black", linestyle="--", label="Linear Trend")

    # Rolling average
    if show_rolling and len(y) >= 5:
        rolling = pd.Series(y).rolling(window=5).mean()
        plt.plot(x, rolling, color="green", linestyle="-", label="Rolling Avg (5)")

    # Outlier annotations
    mean = np.mean(y)
    std = np.std(y)
    for i, val in enumerate(y):
        if val > mean + 2 * std:
            plt.text(x[i], val + 0.5, f"{int(val)}d", ha="center", fontsize=8, color="red")

    plt.title("PR Turnaround Time (Days) – ufs-weather-model", fontsize=14)
    plt.xlabel("PR Number", fontsize=12)
    plt.ylabel("Turnaround Time (Days)", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend()
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
    print("📈 Saved plot to ufs_pr_turnaround_advanced.png")
