import subprocess
import matplotlib.pyplot as plt
import re
import numpy as np

def run_dagger(env_name, expert_file, expert_data, exp_name, n_iter=100):
    print(f"\n{'='*60}")
    print(f"Running DAgger on {env_name} for {n_iter} iterations...")
    print('='*60)
    
    cmd = [
        "python", "rob831/scripts/run_hw1.py",
        "--expert_policy_file", expert_file,
        "--env_name", env_name,
        "--exp_name", exp_name,
        "--n_iter", str(n_iter),
        "--do_dagger",
        "--expert_data", expert_data,
        "--video_log_freq", "-1",
        "--eval_batch_size", "5000"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output)
    
    avg_matches = re.findall(r'Eval_AverageReturn\s*:\s*([\d.]+)', output)
    std_matches = re.findall(r'Eval_StdReturn\s*:\s*([\d.]+)', output)
    
    expert_match = re.search(r'Initial_DataCollection_AverageReturn\s*:\s*([\d.]+)', output)
    expert_return = float(expert_match.group(1)) if expert_match else None
    
    means = [float(m) for m in avg_matches]
    stds = [float(s) for s in std_matches]
    
    return means, stds, expert_return

def run_bc(env_name, expert_file, expert_data, exp_name):
    print(f"\nRunning BC baseline on {env_name}...")
    
    cmd = [
        "python", "rob831/scripts/run_hw1.py",
        "--expert_policy_file", expert_file,
        "--env_name", env_name,
        "--exp_name", exp_name,
        "--n_iter", "1",
        "--expert_data", expert_data,
        "--video_log_freq", "-1",
        "--eval_batch_size", "5000",
        "--num_agent_train_steps_per_iter", "5000"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    avg_match = re.search(r'Eval_AverageReturn\s*:\s*([\d.]+)', output)
    std_match = re.search(r'Eval_StdReturn\s*:\s*([\d.]+)', output)
    expert_match = re.search(r'Initial_DataCollection_AverageReturn\s*:\s*([\d.]+)', output)
    
    bc_mean = float(avg_match.group(1)) if avg_match else None
    bc_std = float(std_match.group(1)) if std_match else None
    expert_return = float(expert_match.group(1)) if expert_match else None
    
    return bc_mean, bc_std, expert_return


ant_dagger_means, ant_dagger_stds, ant_expert = run_dagger(
    "Ant-v2", 
    "rob831/policies/experts/Ant.pkl",
    "rob831/expert_data/expert_data_Ant-v2.pkl",
    "dagger_ant"
)
ant_bc_mean, ant_bc_std, _ = run_bc(
    "Ant-v2",
    "rob831/policies/experts/Ant.pkl", 
    "rob831/expert_data/expert_data_Ant-v2.pkl",
    "bc_ant_baseline"
)

hum_dagger_means, hum_dagger_stds, hum_expert = run_dagger(
    "Humanoid-v2",
    "rob831/policies/experts/Humanoid.pkl",
    "rob831/expert_data/expert_data_Humanoid-v2.pkl", 
    "dagger_humanoid"
)
hum_bc_mean, hum_bc_std, _ = run_bc(
    "Humanoid-v2",
    "rob831/policies/experts/Humanoid.pkl",
    "rob831/expert_data/expert_data_Humanoid-v2.pkl",
    "bc_humanoid_baseline"
)

print("\n" + "="*60)
print("Ant-v2 DAgger Results:")
print("="*60)
for i, (m, s) in enumerate(zip(ant_dagger_means, ant_dagger_stds)):
    print(f"Iter {i}: Mean={m:.2f}, Std={s:.2f}")
print(f"Expert: {ant_expert:.2f}, BC: {ant_bc_mean:.2f}")

print("\n" + "="*60)
print("Humanoid-v2 DAgger Results:")
print("="*60)
for i, (m, s) in enumerate(zip(hum_dagger_means, hum_dagger_stds)):
    print(f"Iter {i}: Mean={m:.2f}, Std={s:.2f}")
print(f"Expert: {hum_expert:.2f}, BC: {hum_bc_mean:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Ant-v2 
ax1 = axes[0]
iters = np.arange(len(ant_dagger_means))
ax1.errorbar(iters, ant_dagger_means, yerr=ant_dagger_stds, marker='o', 
             capsize=5, linewidth=2, markersize=6, label='DAgger', color='steelblue')
ax1.axhline(y=ant_expert, color='green', linestyle='--', linewidth=2, label=f'Expert ({ant_expert:.0f})')
ax1.axhline(y=ant_bc_mean, color='red', linestyle=':', linewidth=2, label=f'BC ({ant_bc_mean:.0f})')
ax1.set_xlabel('DAgger Iteration', fontsize=12)
ax1.set_ylabel('Eval Average Return', fontsize=12)
ax1.set_title('Ant-v2', fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Humanoid-v2 
ax2 = axes[1]
iters = np.arange(len(hum_dagger_means))
ax2.errorbar(iters, hum_dagger_means, yerr=hum_dagger_stds, marker='o',
             capsize=5, linewidth=2, markersize=6, label='DAgger', color='steelblue')
ax2.axhline(y=hum_expert, color='green', linestyle='--', linewidth=2, label=f'Expert ({hum_expert:.0f})')
ax2.axhline(y=hum_bc_mean, color='red', linestyle=':', linewidth=2, label=f'BC ({hum_bc_mean:.0f})')
ax2.set_xlabel('DAgger Iteration', fontsize=12)
ax2.set_ylabel('Eval Average Return', fontsize=12)
ax2.set_title('Humanoid-v2', fontsize=14)
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_dagger_learning_curves.png', dpi=150)
print(f"\nFigure saved to: figure2_dagger_learning_curves.png")
plt.show()