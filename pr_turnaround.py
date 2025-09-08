import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse
from datetime import datetime
from time import sleep

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

LABEL_COLORS = {
    "bug": "#E24A33",
    "enhancement": "#348ABD",
    "documentation": "#988ED5",
    "default": "#4C72B0"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze PR turnaround time across GitHub repos.")
    parser.add_argument("--limit", type=int, default=500, help="Total number of PRs to fetch per repo")
    parser.add_argument("--config", type=str, default="repos.yaml", help="Path to YAML config file")
    return parser.parse_args()

def load_repo_config(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("repos", [])

def fetch_prs_from_url(api_url, state="closed", per_page=100, total_limit=500):
    prs = []
    max_pages = (total_limit + per_page - 1) // per_page
    for page in range(1, max_pages + 1):
        url = f"{api_url}/pulls"
        params = {"state": state, "per_page": per_page, "page": page}
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break
        data = response.json()
        if not data:
            break
        prs.extend(data)
        if len(prs) >= total_limit:
            break
        sleep(0.5)
    return prs[:total_limit]

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

def summarize(df, repo_name):
    merged = df[df["merged"]]
    print(f"\n📊 Summary for {repo_name}")
    print(f"🔁 Total merged PRs: {len(merged)}")
    print(f"⏱ Average turnaround: {merged['turnaround_hours'].mean():.2f} hours")
    print(f"📈 Median turnaround: {merged['turnaround_hours'].median():.2f} hours")
    print(f"🚀 Fastest turnaround: {merged['turnaround_hours'].min():.2f} hours")
    print(f"🐢 Slowest turnaround: {merged['turnaround_hours'].max():.2f} hours")

    print("\n👥 Contributor Summary:")
    by_user = merged.groupby("user")["turnaround_hours"].agg(["count", "mean", "std"]).sort_values("count", ascending=False)
    print(by_user.round(2))

def plot_turnaround(df, repo_name, output_path, show_rolling=True):
    df["turnaround_days"] = df["turnaround_hours"] / 24
    merged = df[df["merged"]].sort_values("number")

    x = merged["number"].values
    y = merged["turnaround_days"].values

    colors = []
    for labels in merged["labels"]:
        label = next((l for l in labels if l in LABEL_COLORS), "default")
        colors.append(LABEL_COLORS[label])

    plt.figure(figsize=(14, 6))
    plt.bar(x, y, color=colors, edgecolor="black")

    coeffs = np.polyfit(x, y, 1)
    trend = np.poly1d(coeffs)
    plt.plot(x, trend(x), color="black", linestyle="--", label="Linear Trend")

    if show_rolling and len(y) >= 5:
        rolling = pd.Series(y).rolling(window=5).mean()
        plt.plot(x, rolling, color="green", linestyle="-", label="Rolling Avg (5)")

    mean = np.mean(y)
    std = np.std(y)
    for i, val in enumerate(y):
        if val > mean + 2 * std:
            plt.text(x[i], val + 0.5, f"{int(val)}d", ha="center", fontsize=8, color="red")

    plt.title(f"PR Turnaround Time (Days) – {repo_name}", fontsize=14)
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
    args = parse_args()
    repos = load_repo_config(args.config)

    for repo in repos:
        name = repo["name"]
        url = repo["url"]
        print(f"\n🔍 Fetching up to {args.limit} PRs from {name}...")
        prs = fetch_prs_from_url(url, total_limit=args.limit)
        print(f"✅ Retrieved {len(prs)} pull requests")
        df = compute_turnaround(prs)
        summarize(df, name)
        df.to_csv(f"{name}_pr_turnaround.csv", index=False)
        plot_turnaround(df, name, output_path=f"{name}_pr_turnaround_advanced.png")
        print(f"📁 Saved CSV: {name}_pr_turnaround.csv")
        print(f"📈 Saved plot: {name}_pr_turnaround_advanced.png")
