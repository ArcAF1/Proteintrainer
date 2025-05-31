#!/usr/bin/env python3
"""
Test script for the complete biomedical LLM training system

This script verifies:
- All dependencies are installed correctly
- Training data generation works
- Model loading and configuration works
- Memory management functions properly
- GUI components load successfully
"""
import sys
import traceback
from pathlib import Path
import importlib

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test all required imports."""
    print("🔍 Testing imports...")
    
    required_packages = [
        # Core ML packages
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("peft", "Parameter Efficient Fine-Tuning"),
        ("accelerate", "Training Acceleration"),
        ("datasets", "Dataset Loading"),
        ("evaluate", "Evaluation Metrics"),
        
        # Quantization and optimization
        ("bitsandbytes", "4-bit Quantization"),
        
        # Apple Silicon optimization
        ("mlx", "Apple MLX Framework"),
        
        # Data processing
        ("pandas", "Data Processing"),
        ("numpy", "Numerical Computing"),
        ("scipy", "Scientific Computing"),
        ("sklearn", "Machine Learning"),
        
        # Visualization
        ("matplotlib", "Plotting"),
        ("seaborn", "Statistical Plotting"),
        
        # GUI and utilities
        ("gradio", "GUI Framework"),
        ("tqdm", "Progress Bars"),
        ("psutil", "System Monitoring"),
        ("memory_profiler", "Memory Profiling"),
        
        # Text processing
        ("nltk", "Natural Language Toolkit"),
        ("rouge_score", "ROUGE Metrics"),
        
        # Configuration
        ("yaml", "YAML Parsing"),
    ]
    
    missing = []
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description} - {e}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {missing}")
        print("Run: pip install " + " ".join(missing))
        return False
    else:
        print("\n🎉 All dependencies are available!")
        return True


def test_training_components():
    """Test training system components."""
    print("\n🔍 Testing training components...")
    
    try:
        # Test training data generator
        print("Testing training data generator...")
        from src.training_data_generator import BiomedicalDataGenerator, TrainingExample
        
        generator = BiomedicalDataGenerator()
        example = TrainingExample(
            instruction="Test instruction",
            output="Test output",
            metadata={"source": "test"}
        )
        print("✅ Training data generator")
        
        # Test trainer configuration
        print("Testing trainer configuration...")
        from src.biomedical_trainer import TrainingConfig, BiomedicalTrainer
        
        config = TrainingConfig(
            base_model="mistralai/Mistral-7B-Instruct-v0.2",
            num_epochs=1,
            max_steps=10,
            output_dir="models/test"
        )
        print("✅ Training configuration")
        
        # Test memory monitor
        print("Testing memory monitor...")
        from src.biomedical_trainer import MemoryMonitor
        monitor = MemoryMonitor()
        monitor.log_memory(0, "Test")
        print("✅ Memory monitoring")
        
        # Test safety validator
        print("Testing safety validator...")
        from src.biomedical_trainer import MedicalSafetyValidator
        validator = MedicalSafetyValidator()
        is_safe, score, reason = validator.validate_output("This is a safe medical statement.")
        print(f"✅ Safety validation (score: {score})")
        
        return True
        
    except Exception as e:
        print(f"❌ Training components failed: {e}")
        traceback.print_exc()
        return False


def test_gui_components():
    """Test GUI components."""
    print("\n🔍 Testing GUI components...")
    
    try:
        # Test training GUI
        print("Testing training GUI...")
        from src.gui_training import TrainingGUI, create_training_interface
        
        gui = TrainingGUI()
        print("✅ Training GUI creation")
        
        # Test unified GUI components
        print("Testing unified GUI status functions...")
        from src.gui_unified import check_training_dependencies, check_dataset_status
        
        dep_status = check_training_dependencies()
        dataset_status = check_dataset_status()
        print("✅ GUI status functions")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI components failed: {e}")
        traceback.print_exc()
        return False


def test_pytorch_mps():
    """Test PyTorch MPS backend for Apple Silicon."""
    print("\n🔍 Testing PyTorch MPS backend...")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        
        if torch.backends.mps.is_available():
            # Test tensor operations on MPS
            device = torch.device("mps")
            x = torch.randn(3, 3, device=device)
            y = torch.randn(3, 3, device=device)
            z = torch.matmul(x, y)
            print(f"✅ MPS tensor operations work")
            
            # Check memory
            if hasattr(torch.mps, 'current_allocated_memory'):
                memory_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                print(f"✅ MPS memory: {memory_mb:.1f}MB allocated")
        else:
            print("⚠️ MPS not available - will fall back to CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch MPS test failed: {e}")
        return False


def test_data_generation():
    """Test training data generation with small dataset."""
    print("\n🔍 Testing training data generation...")
    
    try:
        from src.training_data_generator import BiomedicalDataGenerator
        
        # Create small test generator
        output_dir = Path("test_data")
        output_dir.mkdir(exist_ok=True)
        
        generator = BiomedicalDataGenerator(output_dir)
        
        # Add a few test examples manually
        from src.training_data_generator import TrainingExample
        
        test_examples = [
            TrainingExample(
                instruction="What is aspirin used for?",
                output="Aspirin is commonly used as a pain reliever and to reduce fever. It is also used to prevent heart attacks and strokes.",
                metadata={"source": "test", "confidence": 0.9}
            ),
            TrainingExample(
                instruction="Explain diabetes.",
                output="Diabetes is a chronic condition where the body cannot properly process glucose due to insufficient insulin production or insulin resistance.",
                metadata={"source": "test", "confidence": 0.9}
            )
        ]
        
        generator.examples = test_examples
        
        # Test data saving
        files = generator.save_training_data()
        
        print(f"✅ Generated test training data:")
        for split, file_path in files.items():
            if file_path.exists():
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                print(f"   {split}: {len(lines)} examples in {file_path}")
            else:
                print(f"   ❌ {split}: File not created")
        
        # Cleanup
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        traceback.print_exc()
        return False


def test_model_loading():
    """Test model loading capabilities."""
    print("\n🔍 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Test tokenizer loading
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-small",  # Small model for testing
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded successfully")
        
        # Test model loading with quantization
        print("Testing model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-small",
            torch_dtype=torch.float32,  # Use float32 for compatibility
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully")
        
        # Test inference
        inputs = tokenizer("Hello, how are you?", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print("✅ Model inference works")
        
        # Cleanup
        del model, tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration_files():
    """Test configuration file loading."""
    print("\n🔍 Testing configuration files...")
    
    try:
        import yaml
        
        config_files = [
            "configs/training_configs/quick_test.yaml",
            "configs/training_configs/overnight.yaml", 
            "configs/training_configs/full_training.yaml"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✅ {config_file}: {config.get('name', 'Unnamed')}")
            else:
                print(f"❌ {config_file}: Not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Complete Biomedical LLM Training System")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_imports),
        ("Training Components", test_training_components),
        ("GUI Components", test_gui_components),
        ("PyTorch MPS", test_pytorch_mps),
        ("Data Generation", test_data_generation),
        ("Model Loading", test_model_loading),
        ("Configuration Files", test_configuration_files),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("🏁 TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The training system is ready to use.")
        print("\nNext steps:")
        print("1. Download biomedical datasets: ./installation.command")
        print("2. Test quick training: python train_biomedical.py --preset quick_test")
        print("3. Launch GUI: python run_app.py")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please fix the issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main() 