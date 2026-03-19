import sys
import platform
import torch
import numpy as np

def print_separator():
    print("-" * 80)

def check_version():
    print_separator()
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print_separator()

def check_cuda():
    print("CUDA availability:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
    print_separator()

def test_cpu_tensor():
    print("Testing CPU tensor operations:")
    try:
        # Create and manipulate CPU tensors
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x @ y  # Matrix multiplication
        print("CPU tensor operations: SUCCESS")
        print(f"Sample tensor calculation result:\n{z}")
    except Exception as e:
        print(f"CPU tensor operations: FAILED - {str(e)}")
    print_separator()

def test_cuda_tensor():
    print("Testing CUDA tensor operations:")
    if not torch.cuda.is_available():
        print("CUDA tensor operations: SKIPPED (CUDA not available)")
        return
    
    try:
        # Create and manipulate CUDA tensors
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = x @ y  # Matrix multiplication
        print("CUDA tensor operations: SUCCESS")
        print(f"Sample tensor calculation result:\n{z}")
    except Exception as e:
        print(f"CUDA tensor operations: FAILED - {str(e)}")
    print_separator()

def test_simple_nn():
    print("Testing simple neural network:")
    try:
        # Define a simple neural network
        class SimpleNN(torch.nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                self.fc1 = torch.nn.Linear(10, 5)
                self.fc2 = torch.nn.Linear(5, 2)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # Create model and test forward pass
        model = SimpleNN()
        x = torch.randn(2, 10)
        output = model(x)
        
        print("Neural network forward pass: SUCCESS")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output: {output}")
        
        # Test with CUDA if available
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            output = model(x)
            print("Neural network on CUDA: SUCCESS")
    except Exception as e:
        print(f"Neural network test: FAILED - {str(e)}")
    print_separator()

def test_autograd():
    print("Testing autograd functionality:")
    try:
        x = torch.randn(3, requires_grad=True)
        y = x * 2
        z = y.mean()
        z.backward()
        print("Autograd test: SUCCESS")
        print(f"Gradient: {x.grad}")
    except Exception as e:
        print(f"Autograd test: FAILED - {str(e)}")
    print_separator()

def main():
    print("PyTorch Installation Verification")
    print("=" * 80)
    
    check_version()
    check_cuda()
    test_cpu_tensor()
    test_cuda_tensor()
    test_simple_nn()
    test_autograd()
    
    print("Verification complete!")

if __name__ == "__main__":
    main()