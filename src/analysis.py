import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import os

# Q1 - Load dataset

df = pd.read_csv("../data/marketing_dataset.csv")

print("\n Q1 ")
print("\nFirst 10 rows:\n", df.head(10))
print("\nShape:", df.shape)
print("\nSummary stats:\n", df.describe())


# Q2 — Convert date + sort

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date')
print("\nQ2 ")
print("Min Date:", df['Date'].min(), " | Max Date:", df['Date'].max())


# Q3 — Unique values

print("\nQ3 ")
print("Platforms:", df['Platform'].unique())
print("Projects:", df['Project'].unique())
print("URLs sample:", df['URL'].unique()[:20])
print("Adsets sample:", df['Adset'].unique()[:20])


# Q4 — Filter Google AND visitors > 10000

print("\nQ4 ")
google_high = df[(df['Platform'] == "Google") & (df['Visitors'] > 10000)]
print(google_high)


# Q5 — Create CPL and top 5 expensive adsets

print("\nQ5")
df['CPL'] = np.where(df['Leads'] > 0, df['Spend'] / df['Leads'], np.nan)
top5_cpl = df.sort_values('CPL', ascending=False).head(5)
print(top5_cpl[['Adset','Platform','Spend','Leads','CPL']])


# Q6 — Group by Platform

print("\nQ6")
platform_summary = df.groupby("Platform").agg(
    total_spend=('Spend','sum'),
    total_visitors=('Visitors','sum'),
    total_leads=('Leads','sum'),
    total_closure=('Closure','sum'),
    avg_cpl=('CPL','mean')
).reset_index()
print(platform_summary)
platform_summary.to_csv("../outputs/platform_summary.csv", index=False)


# Q7 — Project with highest SiteVisits

print("\n Q7 ")
proj = df.groupby("Project")['SiteVisits'].sum().idxmax()
print("Project with highest SiteVisits:", proj)


# Q8 — Daily Spend Trend (Plot + Save)

print("\n Q8")
os.makedirs("../outputs/charts", exist_ok=True)

daily_spend = df.groupby(['Date','Platform'])['Spend'].sum().reset_index()
plt.figure(figsize=(12,6))
for p, grp in daily_spend.groupby('Platform'):
    plt.plot(grp['Date'], grp['Spend'], marker='o', label=p)

plt.title("Daily Spend per Platform")
plt.xlabel("Date"); plt.ylabel("Spend")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/charts/daily_spend_per_platform.png")
plt.show()


# Q9 — Funnel Metrics
print("\nQ9")
df['lead_conv_rate'] = df['Leads'] / df['Visitors']
df['sitevisit_conv_rate'] = df['SiteVisits'] / df['Leads'].replace(0,1)
df['closure_conv_rate'] = df['Closure'] / df['SiteVisits'].replace(0,1)

print(df[['Date','Platform','Visitors','Leads','lead_conv_rate','sitevisit_conv_rate','closure_conv_rate']].head())

# Q10 — Anomaly Detection

print("\nQ10")
vis_mean = df['Visitors'].mean()
vis_std = df['Visitors'].std()
lead_mean = df['Leads'].mean()
lead_std = df['Leads'].std()

visitor_anom = df[(df['Visitors'] > vis_mean + 2*vis_std)]
lead_anom = df[(df['Leads'] > lead_mean + 2*lead_std)]

print("Visitor anomalies:", visitor_anom[['Date','Platform','Visitors']].head())
print("Lead anomalies:", lead_anom[['Date','Platform','Leads']].head())

visitor_anom.to_csv("../outputs/visitor_anomalies.csv", index=False)
lead_anom.to_csv("../outputs/lead_anomalies.csv", index=False)


# Q11 — Best Adset per Platform

print(" Q11")
adset_closure = df.groupby(['Platform','Adset'])['Closure'].sum().reset_index()
idx = adset_closure.groupby("Platform")['Closure'].idxmax()
best_adsets = adset_closure.loc[idx]
print(best_adsets)
best_adsets.to_csv("../outputs/best_adsets_by_platform.csv", index=False)


# Q12 — Correlation Heatmap

print("\nQ12")
corr = df[['Spend','Visitors','Leads','SiteVisits','Closure']].corr()
print(corr)

plt.figure(figsize=(7,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.savefig("../outputs/charts/correlation_heatmap.png")
plt.show()

# Q13 — Daily Dashboard CSV

print("\n Q13 ")
daily_dashboard = df.groupby('Date').agg(
    total_spend=('Spend','sum'),
    total_visitors=('Visitors','sum'),
    total_leads=('Leads','sum'),
    total_sitevisits=('SiteVisits','sum'),
    total_closure=('Closure','sum')
).reset_index()
daily_dashboard.to_csv("../outputs/daily_dashboard.csv", index=False)
print(daily_dashboard.head())


# Q14 — Project Funnel Summary

print("\n Q14")
funnel = df.groupby('Project').agg(
    visitors=('Visitors','sum'),
    leads=('Leads','sum'),
    sitevisits=('SiteVisits','sum'),
    closure=('Closure','sum')
).reset_index()

funnel.to_csv("../outputs/project_funnel_summary.csv", index=False)
print(funnel)


# Q15 — Scatter Plot (Spend vs Leads)

print("\n Q15 ")
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Spend', y='Leads', hue='Platform', alpha=0.6)
plt.title("Spend vs Leads")
plt.tight_layout()
plt.savefig("../outputs/charts/spend_vs_leads.png")
plt.show()


# Q16 — Bar Chart: Avg Closure per Platform

print("\nQ16")
avg_closure = df.groupby('Platform')['Closure'].mean().reset_index()
sns.barplot(data=avg_closure, x='Platform', y='Closure')
plt.title("Average Closure by Platform")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("../outputs/charts/avg_closure_by_platform.png")
plt.show()


# Q17 — Daily Visitors for Top 3 Platforms

print("\n Q17 ")
top3 = df.groupby('Platform')['Visitors'].sum().sort_values(ascending=False).head(3).index.tolist()
daily_visitors = df[df['Platform'].isin(top3)].groupby(['Date','Platform'])['Visitors'].sum().reset_index()

plt.figure(figsize=(12,6))
for p in top3:
    temp = daily_visitors[daily_visitors['Platform'] == p]
    plt.plot(temp['Date'], temp['Visitors'], marker='o', label=p)
plt.legend()
plt.title("Daily Visitors - Top 3 Platforms")
plt.tight_layout()
plt.savefig("../outputs/charts/daily_visitors_top3.png")
plt.show()


# Q18 — Platform-wise Dashboard (Spend, Visitors, Leads, Closure)

print("\nQ18 ")
os.makedirs("../outputs/charts/platform_dashboards", exist_ok=True)

for platform in df['Platform'].unique():
    p_df = df[df['Platform'] == platform].groupby('Date').agg(
        spend=('Spend','sum'),
        visitors=('Visitors','sum'),
        leads=('Leads','sum'),
        closure=('Closure','sum')
    ).reset_index()

    fig, axes = plt.subplots(2,2, figsize=(14,10))

    axes[0,0].plot(p_df['Date'], p_df['spend']); axes[0,0].set_title("Spend Trend")
    axes[0,1].plot(p_df['Date'], p_df['visitors']); axes[0,1].set_title("Visitors Trend")
    axes[1,0].plot(p_df['Date'], p_df['leads']); axes[1,0].set_title("Leads Trend")
    axes[1,1].plot(p_df['Date'], p_df['closure']); axes[1,1].set_title("Closure Trend")

    for ax in axes.flatten():
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle(f"{platform} Dashboard", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    file_path = f"../outputs/charts/platform_dashboards/{platform}_dashboard.png"
    fig.savefig(file_path)
    plt.close()

    print(f"Saved dashboard: {file_path}")
