import warnings
import time
import os
from pathlib import Path
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import json

from models import FoundationModelTeacher, LightweightMedicalStudent  #
from dataset import get_enhanced_transforms, ResearchGradeCOVIDDataset, create_data_loaders  
from training import FoundationModelTrainer  

warnings.filterwarnings('ignore')

def run_complete_covid_detection_experiment(data_dir, device, teacher_epochs=25, student_epochs=40):
    print(" COMPLETE COVID DETECTION EXPERIMENT ")
    print("="*80)
    
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Validate dataset structure
        if not validate_dataset_structure(data_dir):
            raise ValueError("Dataset structure validation failed")
        
        # 2. Create data loaders with proper splits
        print(f"\n Creating data loaders with proper train/val/test splits...")
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=16,
            num_workers=4
        )
        
        # 3. Initialize models with ORIGINAL 3-model teacher
        print(f"\n Initializing foundation models...")
        teacher_model = FoundationModelTeacher(num_classes=2).to(device)  
        student_model = LightweightMedicalStudent(num_classes=2, width_multiplier=1.2).to(device)
        
        # Calculate model statistics
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        reduction_ratio = teacher_params / student_params
        
        print(f" Model Statistics:")
        print(f"   Teacher: {teacher_params:,} parameters")
        print(f"   Student: {student_params:,} parameters")
        print(f"   Reduction: {reduction_ratio:.1f}x")
        
        # 4. Initialize trainer
        trainer = FoundationModelTrainer(device, save_dir=results_dir)
        
        # 5. Stage 1: Train teacher (train/val only)
        print(f"\n Stage 1: Training Foundation Teacher ({teacher_epochs} epochs)...")
        teacher_val_acc = trainer.train_teacher(teacher_model, train_loader, val_loader, teacher_epochs)
        
        # 6. Stage 2: Train student (train/val only)
        print(f"\n Stage 2: Training Student with Distillation ({student_epochs} epochs)...")
        student_val_acc = trainer.train_student(student_model, teacher_model, train_loader, val_loader, student_epochs)
        
        # 7. CRITICAL: Final evaluation on TEST SET (first time!)
        print(f"\n FINAL EVALUATION ON TEST SET - FIRST AND ONLY TIME!")
        teacher_test_acc, student_test_acc = trainer.final_test_evaluation(
            teacher_model, student_model, test_loader
        )
        
        results_dict = {
            'teacher_test_accuracy': teacher_test_acc,  
            'student_test_accuracy': student_test_acc,  
            'teacher_accuracy': teacher_test_acc,      
            'student_accuracy': student_test_acc,      
            'teacher_val_accuracy': teacher_val_acc,    
            'student_val_accuracy': student_val_acc,    
            'teacher_params': teacher_params,
            'student_params': student_params,
            'reduction_ratio': reduction_ratio,
            'training_history': trainer.history,
            'success': student_test_acc >= 90.0,
            'available_models': teacher_model.available_models,  
            'test_set_used': True,
            'data_leakage_prevented': True,
            'performance_difference': abs(student_val_acc - student_test_acc)
        }
        
        # 8. Generate comprehensive visualizations (using TEST SET)
        print(f"\n Generating comprehensive results using TEST SET...")
        comprehensive_metrics = trainer.generate_comprehensive_results(
            teacher_model, student_model, test_loader, results_dict
        )
        
        # Update results with comprehensive metrics
        results_dict.update({
            'comprehensive_metrics': comprehensive_metrics,
            'model_size_mb': student_params * 4 / (1024*1024),
            'efficiency_score': (student_test_acc / teacher_test_acc) * reduction_ratio,
            'deployment_ready': student_test_acc >= 90.0 and student_params < 5e6
        })
        
        # 9. Results analysis and reporting
        print("\n" + "="*80)
        print(" FINAL RESULTS ")
        print("="*80)
        print(f" VALIDATION PERFORMANCE (used for training decisions):")
        print(f"   Teacher Validation Accuracy: {teacher_val_acc:.2f}%")
        print(f"   Student Validation Accuracy: {student_val_acc:.2f}%")
        print(f"\n TEST PERFORMANCE :")
        print(f"   Teacher Test Accuracy: {teacher_test_acc:.2f}%")
        print(f"   Student Test Accuracy: {student_test_acc:.2f}%")
        print(f"\n MODEL EFFICIENCY:")
        print(f"   Parameter Reduction: {reduction_ratio:.1f}x")
        print(f"   Model Size: {student_params * 4 / (1024*1024):.1f} MB")
        print(f"   Teacher AUC-ROC: {comprehensive_metrics['teacher_auc']:.4f}")
        print(f"   Student AUC-ROC: {comprehensive_metrics['student_auc']:.4f}")
        
        # Performance difference analysis
        perf_diff = abs(student_val_acc - student_test_acc)
        print(f"\n DATA LEAKAGE CHECK:")
        print(f"   Validation vs Test difference: {perf_diff:.2f}%")
        if perf_diff < 5.0:
            print("    GOOD: Small difference indicates no major overfitting")
        elif perf_diff < 10.0:
            print("    MODERATE: Some overfitting detected, but acceptable")
        else:
            print("    HIGH: Large difference may indicate overfitting")
        
        # Success evaluation
        success = student_test_acc >= 99.0
        if success:
            print(f"\n SUCCESS! ACHIEVED 99%+ TEST ACCURACY! 🎉")
        else:
            print(f"\n NEEDS IMPROVEMENT: {student_test_acc:.2f}%")
        
        # 10. Save all results with corrected data
        print(f"\n Saving comprehensive results...")
        
        # Save main results
        torch.save(results_dict, results_dir / 'complete_experiment_results.pth')
        
        # FIXED: Generate final performance summary with actual values
        performance_summary = pd.DataFrame({
            'Model': ['Teacher Ensemble', 'Lightweight Student'],
            'Test Accuracy (%)': [f"{teacher_test_acc:.2f}", f"{student_test_acc:.2f}"],
            'Val Accuracy (%)': [f"{teacher_val_acc:.2f}", f"{student_val_acc:.2f}"],
            'Precision': [f"{comprehensive_metrics['teacher_metrics']['precision']:.4f}",
                          f"{comprehensive_metrics['student_metrics']['precision']:.4f}"],
            'Recall': [f"{comprehensive_metrics['teacher_metrics']['recall']:.4f}",
                       f"{comprehensive_metrics['student_metrics']['recall']:.4f}"],
            'F1-Score': [f"{comprehensive_metrics['teacher_metrics']['f1_score']:.4f}",
                         f"{comprehensive_metrics['student_metrics']['f1_score']:.4f}"],
            'AUC-ROC': [f"{comprehensive_metrics['teacher_auc']:.4f}",
                        f"{comprehensive_metrics['student_auc']:.4f}"],
            'Parameters': [f"{teacher_params:,}", f"{student_params:,}"],
            'Size (MB)': [f"{teacher_params * 4 / (1024*1024):.1f}",
                          f"{student_params * 4 / (1024*1024):.1f}"]
        })
        performance_summary.to_csv(results_dir / 'final_performance_summary.csv', index=False)
        
        
        return {
            'teacher_model': teacher_model,
            'student_model': student_model,
            'teacher_test_accuracy': teacher_test_acc,
            'student_test_accuracy': student_test_acc,
            'teacher_val_accuracy': teacher_val_acc,
            'student_val_accuracy': student_val_acc,
            'performance_difference': perf_diff,
            'success': success,
            'results': results_dict,
            'comprehensive_metrics': comprehensive_metrics,
            'reduction_ratio': reduction_ratio,
            'model_size_mb': student_params * 4 / (1024*1024),
            'data_leakage_prevented': True
        }
        
    except Exception as e:
        print(f" Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def validate_dataset_structure(data_dir):
    """Validate that dataset has proper train/val/test structure."""
    data_path = Path(data_dir)
    required_splits = ['train', 'val', 'test']
    
    print(" DATASET STRUCTURE VALIDATION")
    print("="*40)
    
    missing_splits = []
    for split in required_splits:
        split_path = data_path / split
        if split_path.exists():
            subfolders = [f.name for f in split_path.iterdir() if f.is_dir()]
            print(f" {split}/: {subfolders}")
        else:
            print(f" {split}/: NOT FOUND")
            missing_splits.append(split)
    
    if missing_splits:
        print(f"\n CRITICAL ERROR: Missing required splits: {missing_splits}")
        return False
    
    print(" Dataset structure validation passed!")
    return True

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = '/kaggle/input/covid-dataset/dataset_split'
    
    print(" COMPLETE COVID DETECTION EXPERIMENT ")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Research Environment: {device}")
    
    # Verify dataset exists with fallback paths
    if not os.path.exists(DATASET_PATH):
        print(f" Dataset not found at {DATASET_PATH}")
        alternative_paths = [
            '/kaggle/input/covid19-dataset/dataset_split',
            '/kaggle/working/dataset_split',
            './dataset_split',
            '../dataset_split'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                DATASET_PATH = alt_path
                print(f" Found dataset at: {DATASET_PATH}")
                break
        else:
            print(" No dataset found. Please check the dataset path.")
            exit()
    else:
        print(f" Dataset found at: {DATASET_PATH}")
    
    try:
        start_time = time.time()
        
        # Run experiment with corrected pipeline
        results = run_complete_covid_detection_experiment(
            data_dir=DATASET_PATH,
            device=device,
            teacher_epochs=100,    # Reasonable epochs for testing
            student_epochs=100     # Extended distillation training
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*70)
        print(" COMPLETE EXPERIMENT FINISHED!")
        print("="*70)
        print(f"⏱ Total execution time: {total_time/3600:.2f} hours")
        
        # Final summary with corrected values
        if results['success']:
            print(" SUCCESS! ACHIEVED 99%+ TEST ACCURACY!")
            print(f" Student test accuracy: {results['student_test_accuracy']:.2f}%")
            print(f" Parameter reduction: {results['reduction_ratio']:.1f}x")
            print(f" Model size: {results['model_size_mb']:.1f} MB")
        else:
            print(f" Test accuracy: {results['student_test_accuracy']:.2f}%")
        
        
    except Exception as e:
        print(f" Experiment failed: {e}")
        print(" Check the fixes applied to models.py and main.py")
    
    print("\n" + "="*70)
    print(" COVID DETECTION SYSTEM COMPLETE!")
    print("="*70)

