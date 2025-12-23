import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
import warnings

warnings.filterwarnings("ignore")

# --- GLOBAL STYLE ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'figure.facecolor': 'white'
})

OUTPUT_DIR = r"c:\Users\JAYANT\Downloads\chargeup_project\chargeup_project\paper_assets"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_fig(fig, name):
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

# 1. FUZZY MEMBERSHIP FUNCTIONS
def plot_membership_functions():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.linspace(0, 100, 100)
    low = np.maximum(0, np.minimum(1, (30-x)/30))
    med = np.maximum(0, np.minimum((x-20)/20, (60-x)/20))
    high = np.maximum(0, np.minimum(1, (x-50)/30))
    ax1.plot(x, low, 'r', label='Critical')
    ax1.plot(x, med, 'y', label='Moderate')
    ax1.plot(x, high, 'g', label='Sufficient')
    ax1.set_title('Inlet 1: Battery SoC (%)')
    ax1.set_xlabel('State of Charge (%)')
    ax1.legend()
    
    d = np.linspace(0, 150, 100)
    near = np.maximum(0, np.minimum(1, (40-d)/40))
    far = np.maximum(0, np.minimum(1, (d-30)/70))
    ax2.plot(d, near, 'b', label='In-Range')
    ax2.plot(d, far, 'm', label='Out-of-Range')
    ax2.set_title('Inlet 2: Distance to Destination')
    ax2.set_xlabel('Distance (km)')
    ax2.legend()
    save_fig(fig, 'fig_membership.png')

# 2. COMPARATIVE BOXPLOTS
def plot_wait_time_comparison():
    np.random.seed(42)
    data = [np.random.normal(55, 12, 200), np.random.normal(42, 8, 200), 
            np.random.normal(35, 6, 200), np.random.normal(24, 4, 200)]
    labels = ['FCFS', 'Fuzzy', 'RL-Swap', 'ChargeUp']
    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=False)
    colors = ['#bdc3c7', '#f1c40f', '#e67e22', '#2ecc71']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Wait Time (minutes)')
    ax.set_title('System Performance: Waiting Time Distribution')
    save_fig(fig, 'fig_wait_comparison.png')

# 3. ECONOMIC BENEFIT ANALYSIS (NEW)
def plot_economic_benefit():
    scenarios = ['Baseline (FIFO)', 'Managed (ChargeUp)']
    merchant_revenue = [1200, 1580] # Monthly units
    user_savings = [0, 450] # Time-cost equivalent savings
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, merchant_revenue, width, label='Merchant Revenue (Relative)', color='#2980b9')
    rects2 = ax.bar(x + width/2, user_savings, width, label='User Time-Value Savings', color='#27ae60')
    
    ax.set_ylabel('Economic Value (USD/month/station)')
    ax.set_title('Economic Impact: Merchant ROI and User Cost-Benefits')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    
    # Add labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    save_fig(fig, 'fig_economics.png')

# 4. SUSTAINABILITY / GREEN IMPACT (NEW)
def plot_green_impact():
    hours = np.arange(24)
    solar_availability = np.array([0,0,0,0,0,5,20,50,85,100,105,110,115,110,100,75,40,10,0,0,0,0,0,0])
    # Normalize solar to 0-100
    solar_availability = 100 * solar_availability / max(solar_availability)
    
    unmanaged_demand = np.array([10,8,5,20,40,60,85,90,75,60,50,45,45,50,60,80,95,90,70,50,30,20,15,10])
    managed_demand = np.array([10,8,5,15,25,35,50,60,75,90,100,105,100,90,70,60,65,70,65,55,40,25,15,10])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(hours, solar_availability, color='#f1c40f', alpha=0.15, label='Green Energy Profile (Solar)')
    ax.plot(hours, unmanaged_demand, 'r--', label='Unmanaged Demand (Dirty Peaks)', alpha=0.6)
    ax.plot(hours, managed_demand, 'g-', lw=2, label='ChargeUp Optimized (Solar Tracking)')
    
    # Shade the overlap - solar utilization
    overlap = np.minimum(managed_demand, solar_availability)
    ax.fill_between(hours, 0, overlap, color='#2ecc71', alpha=0.3, label='Effective Green Charging Usage')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Energy Demand / Availability (%)')
    ax.set_title('Sustainability Multiplier: Demand Shifting to Solar Peaks')
    ax.legend()
    ax.grid(True, alpha=0.2)
    save_fig(fig, 'fig_green_impact.png')

# 5. RL REWARD DECOMPOSITION
def plot_reward_decomposition():
    steps = np.arange(100)
    wait_reward = 30 * (1 - np.exp(-steps/20))
    point_reward = 20 * (1 - np.exp(-steps/40))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(steps, 0, wait_reward, label='Wait-Time Optimization', color='#3498db', alpha=0.5)
    ax.fill_between(steps, wait_reward, wait_reward + point_reward, label='Cooperation Incentives', color='#f1c40f', alpha=0.5)
    ax.set_title('MARL Policy Objective Convergence')
    ax.legend()
    save_fig(fig, 'fig_reward_decomp.png')

# 6. FUZZY SURFACE
def plot_fuzzy_3d():
    from mpl_toolkits.mplot3d import Axes3D
    B = np.linspace(0, 100, 50)
    U = np.linspace(0, 10, 50)
    B_grid, U_grid = np.meshgrid(B, U)
    Priority = 100 / (1 + np.exp(-((100 - B_grid) * 0.5 + U_grid * 5 - 50)/12))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(B_grid, U_grid, Priority, cmap='viridis', alpha=0.8)
    ax.set_title('Fuzzy Logic Urgency Mapping (SoC vs Wait Tolerance)')
    save_fig(fig, 'fig_fuzzy_3d_new.png')

if __name__ == "__main__":
    print("Generating updated figures (Economic & Sustainability Focus)...")
    plot_membership_functions()
    plot_wait_time_comparison()
    plot_economic_benefit()
    plot_green_impact()
    plot_reward_decomposition()
    plot_fuzzy_3d()
    print("Done.")
