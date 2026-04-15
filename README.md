# Sim2Sim_Lab: Domain Shift Evaluation for LIBERO

Sim2Sim_Lab is an experimental framework designed for zero-shot domain shift evaluation on the [LIBERO Robot Manipulation Dataset](https://libero-project.github.io/).

Because the underlying Robosuite environment used by LIBERO performs a "hard reset" (completely rebuilding MuJoCo simulation objects from XML) every time `env.reset()` is called, standard modifications to model parameters are lost between episodes. This project solves this issue by using dynamic injection and monkey-patching techniques to intercept the environment reset process at runtime. It automatically reapplies physical and visual domain shift configurations, enabling efficient batch testing of policy robustness.



## 📥 Installation: Cloning the Repository

When pulling this project from GitHub, it is **crucial** to clone it recursively to ensure all submodules (such as LIBERO and OpenPI dependencies) are properly initialized and downloaded. 

Please use the following command:

```bash
# Clone the repository and recursively initialize all submodules
git clone --recursive [https://github.com/your-username/Sim2Sim_Lab.git](https://github.com/your-username/Sim2Sim_Lab.git)```
(Note: If you have already cloned the repository without the --recursive flag, you can fetch the submodules by running git submodule update --init --recursive inside the project folder).

🌟 Core Features & Supported Domain Shifts
This project allows you to modify the MuJoCo environment at runtime using simple YAML configuration files. Currently, the following types of domain shifts are supported:

Lighting: Diffuse/specular intensity scaling, ambient light adjustments, light position/direction offsets, color temperature shifts (warm/cool), and shadow toggling.

Camera: Camera position offsets, Euler rotation offsets, and Field of View (FOV) scaling.

Friction: Global friction scaling and local friction modifications based on geometry (Geom) name keywords.

Material & Optics: Global specular, shininess, and reflectance scaling, as well as local material property overrides based on keywords.

Geom RGBA: Geometry color replacement or transparency adjustments based on keywords.

🛠️ Environment Configuration (Docker)
We recommend using Docker (or cloud container services like RunPod) for environment isolation and configuration.

1. Starting the Container via Docker Compose

Important: Before starting the Docker container, you must navigate to the root directory of the project. We utilize docker-compose to seamlessly manage volumes, ports, and runtime execution.

```Bash
# Step 1: Navigate to the project root directory
cd Sim2Sim_Lab

# Step 2: Build and start the container in detached mode using docker-compose
docker compose -f docker/docker-compose.yml up -d```
2. Entrypoint Script Details

Once the container is launched via docker-compose, the included entrypoint.sh script automatically performs the following initialization steps:

Creates a symlink from /app to your actual working directory (APP_OVERRIDE) to ensure absolute path compatibility.

Automatically sets the PYTHONPATH to include the source code for OpenPI and LIBERO.

Starts JupyterLab (port 8888) and TensorBoard (port 6006) in the background for real-time monitoring of training and testing logs.

🚀 Running Experiments & Selecting Models
This project is compatible with and utilizes the OpenPI (Physical Intelligence) framework interface to load and deploy models. During evaluation, the script starts a Policy Server, and the LIBERO environment requests actions via network calls.

1. Configuring the Test Model

In the run_batch.sh script, you can specify the models to test by modifying the CHECKPOINTS array. The format is "policy_config|checkpoint_dir":

```Bash
CHECKPOINTS=(
    # Format: "OpenPI_Policy_Config_Name|Path_to_Model_Weights"
    "pi05_libero|gs://openpi-assets/checkpoints/pi05_libero"
    "my_custom_policy|/app/data/checkpoints/my_custom_policy_v1"
)```
2. Starting Batch Tests

After configuring the task suites (TASK_SUITES) and the domain configuration directory (DOMAIN_CONFIG_DIR) in run_batch.sh, simply run:

```Bash
# Default is 20 trials per task
bash /app/eval/run_batch.sh

# Or override the number of trials via environment variables
NUM_TRIALS=50 bash /app/eval/run_batch.sh```
The script will automatically iterate through all Checkpoints, Task Suites, and Domain Shift levels (weak, medium, strong). Upon completion, it will generate a summarized CSV table in the RESULTS_ROOT directory.

🎨 How to Implement a New Domain Shift
To add a new domain shift test, simply create a new YAML file in the domain_configs/ directory. The system supports both flat formats and multi-level formats (levels). We recommend using the multi-level format to automatically run ablation studies (weak, medium, strong) via run_batch.sh.

YAML Configuration Template Example (domain_configs/camera_shift.yaml)

```YAML
levels:
  weak:
    camera:
      shifts:
        - name: "agentview"
          pos_offset: [0.05, 0.0, 0.0]        # Offset X-axis by 5cm
          euler_offset_deg: [0.0, 2.0, 0.0]   # Pitch rotation by 2 degrees
          fovy_offset: 2.0                    # Increase FOV by 2 degrees
  
  medium:
    camera:
      shifts:
        - name: "agentview"
          pos_offset: [0.1, 0.0, 0.0]
          euler_offset_deg: [0.0, 5.0, 0.0]
          fovy_offset: 5.0

  strong:
    camera:
      shifts:
        - name: "agentview"
          pos_offset: [0.2, 0.05, 0.0]
          euler_offset_deg: [0.0, 10.0, 0.0]
          fovy_offset: 10.0```
Supported Configuration Fields Reference

lighting: diffuse_scale, specular_scale, ambient_scale, direction_offset, color_shift, castshadow.

friction: global_scale, geom_friction_shifts (includes name_contains and friction_scale).

material: global_specular_scale, material_shifts (includes name_contains, specular, rgba, etc.).

geom_rgba_shifts: Replace rgba or adjust rgba_scale based on name_contains.

As long as the YAML file is placed in DOMAIN_CONFIG_DIR, run_batch.sh will automatically parse and execute it.

🔮 Future Extensions: Supporting Other Open-Source Models
Currently, this framework primarily conducts policy inference via OpenPI's Server-Client architecture. We plan to add testing support for other mainstream open-source vision-action models (such as OpenVLA, Octo, etc.) in the future.

How to integrate other models now:
You can integrate other open-source models (like OpenVLA) by modifying their source code to wrap them into a Python interface compatible with this framework's calling conventions. Simply introduce them as third-party libraries in the /app/third_party/ directory. As long as your model can receive the obs (containing images and robot states) returned by the LIBERO environment and output the corresponding action, it can be seamlessly integrated into this project's Domain Shift evaluation pipeline.
