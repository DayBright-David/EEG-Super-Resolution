#!/bin/bash

# EEG Super-Resolution Experiment Runner
# This script provides a convenient way to run various experiments

set -e  # Exit on any error

# Default configuration
PYTHON_EXEC="python"
CONFIG_FILE="configs/default_config.yaml"
DEVICE="cuda"
NUM_GPUS=1

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if required dependencies are installed
check_dependencies() {
    print_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check PyTorch
    python -c "import torch" 2>/dev/null || {
        print_error "PyTorch not found. Please install PyTorch"
        exit 1
    }
    
    # Check if GPU is available (if CUDA device specified)
    if [[ $DEVICE == "cuda" ]]; then
        python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
            print_warning "CUDA not available, falling back to CPU"
            DEVICE="cpu"
        }
    fi
    
    print_success "Dependencies check passed"
}

# Function to create necessary directories
setup_directories() {
    print_info "Setting up directories..."
    
    mkdir -p checkpoints
    mkdir -p logs
    mkdir -p results
    mkdir -p dataset
    
    print_success "Directories created"
}

# Function to run pretraining
run_pretraining() {
    local subjects=${1:-"all"}
    
    print_info "Starting pretraining for subjects: $subjects"
    
    if [[ $subjects == "all" ]]; then
        # Run for all subjects (1-35)
        for subject in {1..35}; do
            print_info "Pretraining subject $subject..."
            
            if [[ $NUM_GPUS -gt 1 ]]; then
                torchrun --nproc_per_node=$NUM_GPUS scripts/run_labram_pretraining.py \
                    --output_dir ./checkpoints/subject_${subject} \
                    --log_dir ./logs/pretrain/subject_${subject} \
                    --user_id $subject \
                    --device $DEVICE
            else
                python scripts/run_labram_pretraining.py \
                    --output_dir ./checkpoints/subject_${subject} \
                    --log_dir ./logs/pretrain/subject_${subject} \
                    --user_id $subject \
                    --device $DEVICE
            fi
        done
    else
        # Run for specific subject
        print_info "Pretraining subject $subjects..."
        
        if [[ $NUM_GPUS -gt 1 ]]; then
            torchrun --nproc_per_node=$NUM_GPUS scripts/run_labram_pretraining.py \
                --output_dir ./checkpoints/subject_${subjects} \
                --log_dir ./logs/pretrain/subject_${subjects} \
                --user_id $subjects \
                --device $DEVICE
        else
            python scripts/run_labram_pretraining.py \
                --output_dir ./checkpoints/subject_${subjects} \
                --log_dir ./logs/pretrain/subject_${subjects} \
                --user_id $subjects \
                --device $DEVICE
        fi
    fi
    
    print_success "Pretraining completed"
}

# Function to run evaluation
run_evaluation() {
    local checkpoint_dir=${1:-"./checkpoints"}
    local epoch=${2:-"1999"}
    
    print_info "Running evaluation with checkpoint epoch $epoch"
    
    python scripts/test_pretrain_on_pretrain_data.py \
        --checkpoint_path $checkpoint_dir \
        --epochs $epoch \
        --device $DEVICE
    
    python scripts/test_pretrain_on_finetune_data.py \
        --checkpoint_path $checkpoint_dir \
        --epochs $epoch \
        --sub 1 \
        --device $DEVICE
    
    print_success "Evaluation completed"
}

# Function to run SSVEP classification experiments
run_ssvep_experiments() {
    local checkpoint_dir=${1:-"./checkpoints"}
    local epoch=${2:-"1999"}
    
    print_info "Running SSVEP classification experiments"
    
    # Standard TDCA baseline
    print_info "Running baseline TDCA..."
    python scripts/train_ssvep_tdca_test_channel_attack_multi_time_n_fold.py \
        --pretrain_checkpoint_path $checkpoint_dir \
        --checkpoint_epoch $epoch \
        --device $DEVICE \
        --output_dir ./results/baseline
    
    # With super-resolution
    print_info "Running TDCA with super-resolution..."
    python scripts/train_ssvep_tdca_test_channel_attack_multi_time_n_fold.py \
        --pretrain_checkpoint_path $checkpoint_dir \
        --checkpoint_epoch $epoch \
        --use_super_resolution \
        --device $DEVICE \
        --output_dir ./results/super_resolution
    
    print_success "SSVEP experiments completed"
}

# Function to run quick demo
run_demo() {
    print_info "Running quick demo..."
    
    python examples/quick_start.py
    
    print_success "Demo completed"
}

# Function to display help
show_help() {
    echo "EEG Super-Resolution Experiment Runner"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup              Set up directories and check dependencies"
    echo "  pretrain [SUBJECT] Run pretraining (default: all subjects)"
    echo "  evaluate [DIR] [EPOCH] Run evaluation (default: ./checkpoints, epoch 1999)"
    echo "  ssvep [DIR] [EPOCH]    Run SSVEP experiments"
    echo "  demo               Run quick demonstration"
    echo "  all                Run complete pipeline"
    echo "  help               Show this help message"
    echo ""
    echo "Options:"
    echo "  --device DEVICE    Device to use (cuda/cpu, default: cuda)"
    echo "  --gpus NUM         Number of GPUs to use (default: 1)"
    echo "  --config FILE      Configuration file (default: configs/default_config.yaml)"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 pretrain 1"
    echo "  $0 pretrain all"
    echo "  $0 evaluate ./checkpoints 1999"
    echo "  $0 demo"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Main command processing
case ${1:-help} in
    setup)
        check_dependencies
        setup_directories
        ;;
    pretrain)
        check_dependencies
        setup_directories
        run_pretraining ${2:-all}
        ;;
    evaluate)
        check_dependencies
        run_evaluation ${2:-./checkpoints} ${3:-1999}
        ;;
    ssvep)
        check_dependencies
        run_ssvep_experiments ${2:-./checkpoints} ${3:-1999}
        ;;
    demo)
        check_dependencies
        run_demo
        ;;
    all)
        check_dependencies
        setup_directories
        run_pretraining all
        run_evaluation
        run_ssvep_experiments
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 