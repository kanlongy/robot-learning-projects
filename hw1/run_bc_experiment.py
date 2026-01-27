import subprocess
import matplotlib.pyplot as plt
import re

steps_list = [100, 500, 1000, 2000, 5000, 10000]
means = []
stds = []


for steps in steps_list:
    print(f"\n{'='*50}")
    print(f"Running experiment with {steps} training steps...")
    print('='*50)
    
    cmd = [
        "python", "rob831/scripts/run_hw1.py",
        "--expert_policy_file", "rob831/policies/experts/Ant.pkl",
        "--env_name", "Ant-v2",
        "--exp_name", f"bc_ant_steps{steps}",
        "--n_iter", "1",
        "--expert_data", "rob831/expert_data/expert_data_Ant-v2.pkl",
        "--video_log_freq", "-1",
        "--eval_batch_size", "5000",
        "--num_agent_train_steps_per_iter", str(steps)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    avg_match = re.search(r'Eval_AverageReturn\s*:\s*([\d.]+)', output)
    std_match = re.search(r'Eval_StdReturn\s*:\s*([\d.]+)', output)
    
    if avg_match and std_match:
        mean_val = float(avg_match.group(1))
        std_val = float(std_match.group(1))
        means.append(mean_val)
        stds.append(std_val)
        print(f"Steps: {steps}, Mean: {mean_val:.2f}, Std: {std_val:.2f}")
    else:
        print(f"Failed to parse results for steps={steps}")
        print(output)

print("\n" + "="*60)
print("Results Summary:")
print("="*60)
print(f"{'Steps':<10} {'Mean Return':<15} {'Std Return':<15}")
print("-"*40)
for s, m, std in zip(steps_list, means, stds):
    print(f"{s:<10} {m:<15.2f} {std:<15.2f}")

plt.figure(figsize=(10, 6))
plt.errorbar(steps_list, means, yerr=stds, marker='o', capsize=5, 
             linewidth=2, markersize=8, color='steelblue', ecolor='gray')
plt.xlabel('Number of Training Steps', fontsize=12)
plt.ylabel('Eval Average Return', fontsize=12)
plt.title('BC Performance vs Training Steps (Ant-v2)', fontsize=14)
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_bc_training_steps.png', dpi=150)
print(f"\nFigure saved to: figure1_bc_training_steps.png")
plt.show()