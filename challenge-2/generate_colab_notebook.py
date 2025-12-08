#!/usr/bin/env python3
"""
Generate Colab-ready notebook by injecting modules into TRAIN.ipynb template
Simple approach: read template, inject modules cell, save as COLAB_READY.ipynb
"""
import json
from pathlib import Path


def read_module_file(filepath):
    """Read a module file and return its content"""
    with open(filepath, 'r') as f:
        return f.read()


def create_modules_cell():
    """Create the cell containing all embedded modules
    
    Merges dependencies into consumer files to avoid import issues.
    Each dependency is included only once, before the code that uses it.
    """
    utils_dir = Path('utils')
    
    # Define what to include (dependencies merged inline, no duplication)
    modules_to_include = [
        # Config - standalone
        ('config.py', utils_dir / 'config.py'),
        
        # Automated augmentation (used by dataset)
        ('automated_augmentation.py', utils_dir / 'automated_augmentation.py'),
        
        # Dataset (uses automated_augmentation defined above)
        ('dataset.py', utils_dir / 'dataset.py'),
        
        # Normalization (used by models)
        ('normalization.py', utils_dir / 'normalization.py'),
        
        # Models (uses normalization defined above)
        ('models.py', utils_dir / 'models.py'),
        
        # Advanced optimizers (used by trainer)
        ('advanced_optimizers.py', utils_dir / 'advanced_optimizers.py'),
        
        # MixUp (used by trainer)
        ('mixup.py', utils_dir / 'mixup.py'),
        
        # Trainer (uses advanced_optimizers and mixup defined above)
        ('trainer.py', utils_dir / 'trainer.py'),
        
        # Evaluation (uses dataset functions defined above)
        ('evaluation.py', utils_dir / 'evaluation.py'),
        
        # Visualization - standalone
        ('visualization.py', utils_dir / 'visualization.py'),
        
        # Data cleaning - standalone
        ('data_cleaning.py', utils_dir / 'data_cleaning.py'),
        
        # Outlier detection - standalone
        ('outlier_detection.py', utils_dir / 'outlier_detection.py'),
    ]
    
    # Build module code - each file included exactly once
    module_code_parts = ["#@title 📦 Module Files (click to expand)\n"]
    module_code_parts.append("# All modules embedded - no imports needed\n\n")
    
    for module_name, module_path in modules_to_include:
        module_content = read_module_file(module_path)
        module_code_parts.append(f"\n# {'='*60}\n")
        module_code_parts.append(f"# {module_name}\n")
        module_code_parts.append(f"# {'='*60}\n\n")
        module_code_parts.append(module_content)
        module_code_parts.append("\n\n")
    
    module_code_parts.append("print('✅ All modules loaded')")
    
    return {
        "cell_type": "code",
        "metadata": {
            "cellView": "form"  # Collapsed by default
        },
        "source": module_code_parts,
        "execution_count": None,
        "outputs": []
    }


def create_config_cell():
    """Create the cell containing config.yaml content"""
    # Use optimized config instead of default config
    config_file = 'config_optimized.yaml' if Path('config_optimized.yaml').exists() else 'config.yaml'
    
    with open(config_file, 'r') as f:
        config_yaml = f.read()
    
    # Build source lines properly, escaping the YAML content
    source_lines = [
        f"#@title 📝 {config_file} (click to expand)\n",
        "config_yaml_content = r'''" + config_yaml + "'''\n",
        "\n",
        "# Write config to file\n",
        "with open('config.yaml', 'w') as f:\n",
        "    f.write(config_yaml_content)\n",
        "\n",
        f"print('✅ config.yaml created (from {config_file})')"
    ]
    
    return {
        "cell_type": "code",
        "metadata": {
            "cellView": "form"  # Collapsed by default
        },
        "source": source_lines,
        "execution_count": None,
        "outputs": []
    }


def inject_modules_into_template():
    """Read TRAIN.ipynb template and inject modules"""
    
    # Read template notebook
    with open('TRAIN.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Add title cell at the beginning
    title_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🧬 Tissue Classification - Colab Ready\n",
            "\n",
            "**Self-contained notebook** - All modules embedded below\n",
            "\n",
            "Generated from `TRAIN.ipynb` template with modules injected\n",
            "\n",
            "---"
        ]
    }
    
    # Add Colab mount cell
    mount_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Mount Google Drive (for Colab)\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive')\n",
            "    \n",
            "    # Change to your data directory\n",
            "    import os\n",
            "    os.chdir('/content/drive/MyDrive/Challenge2')\n",
            "    print(f\"✅ Working directory: {os.getcwd()}\")\n",
            "except:\n",
            "    print(\"ℹ️  Not running in Colab, using current directory\")"
        ],
        "execution_count": None,
        "outputs": []
    }
    
    # Add dependencies install cell
    deps_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Install dependencies\n",
            "!pip install -q pyyaml tensorboard scikit-learn tqdm\n",
            "print('✅ Dependencies installed')"
        ],
        "execution_count": None,
        "outputs": []
    }
    
    # Create injected cells
    config_cell = create_config_cell()
    modules_cell = create_modules_cell()
    
    # Insert cells at the beginning
    notebook['cells'] = [
        title_cell,
        mount_cell,
        deps_cell,
        config_cell,
        modules_cell
    ] + notebook['cells']
    
    # Update metadata for Colab
    notebook['metadata'] = {
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    }
    
    return notebook


def main():
    """Generate Colab-ready notebook from template"""
    print("🔨 Generating Colab notebook from TRAIN.ipynb template...")
    
    # Check template exists
    if not Path('TRAIN.ipynb').exists():
        print("❌ Error: TRAIN.ipynb template not found")
        return
    
    # Generate notebook
    notebook = inject_modules_into_template()
    
    # Save to file
    output_path = Path('COLAB_READY.ipynb')
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Notebook generated: {output_path}")
    print(f"   Total cells: {len(notebook['cells'])}")
    print(f"   Template cells: {len(notebook['cells']) - 5}")
    print(f"   Injected cells: 5 (title + mount + deps + config + modules)")
    print(f"\n📤 Upload to Google Colab and run!")


if __name__ == '__main__':
    main()
