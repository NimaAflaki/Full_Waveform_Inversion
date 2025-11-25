from stride import *
import numpy as np
import matplotlib.pyplot as plt

def main():
    vp_NN = np.load('NN14.npy')
    vp_homo = np.load('Homo14.npy')
    gt = np.load('data/gt_rect.npy').T
    mask = np.load('mask.npy')
    mask2 = np.load('mask2.npy')
    NN_guess = np.load('data/pred_rect.npy').T
    NN_guess = np.where(NN_guess == 0, 2500, NN_guess)
    mask_NN = NN_guess < 2000
    mask3 = mask & mask_NN

    # Create figure with subplots for better comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Ground Truth
    im1 = axes[0, 0].imshow(gt, cmap='viridis', vmin=1400, vmax=1700)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Z')
    plt.colorbar(im1, ax=axes[0, 0], label='Vp (m/s)')

    # Plot 2: Mask
    im2 = axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Binary Mask')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes[0, 1], label='Mask Value')

    # Plot 3: NN Prediction
    im3 = axes[0, 2].imshow(vp_NN, cmap='viridis', vmin=1400, vmax=1700)
    axes[0, 2].set_title('FWI with NN Starting Model')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Z')
    plt.colorbar(im3, ax=axes[0, 2], label='Vp (m/s)')

    # Plot 4: Homogeneous Starting Model Result
    im4 = axes[1, 0].imshow(vp_homo, cmap='viridis', vmin=1400, vmax=1700)
    axes[1, 0].set_title('FWI with Homogeneous Starting Model')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Z')
    plt.colorbar(im4, ax=axes[1, 0], label='Vp (m/s)')

    # Calculate differences
    diff_homo = vp_homo - gt
    diff_NN = vp_NN - gt

    # Normalize to GT
    norm_diff_homo = diff_homo / gt
    norm_diff_NN = diff_NN / gt

    # Plot 5: Normalized Difference (Homo)
    im5 = axes[1, 1].imshow(norm_diff_homo, cmap='RdBu_r', vmax=0.05, vmin=-0.05)
    axes[1, 1].set_title('Normalized Error: Homogeneous')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Z')
    plt.colorbar(im5, ax=axes[1, 1], label='(Pred - GT) / GT')

    # Plot 6: Normalized Difference (NN)
    im6 = axes[1, 2].imshow(norm_diff_NN, cmap='RdBu_r', vmax=0.05, vmin=-0.05)
    axes[1, 2].set_title('Normalized Error: NN')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Z')
    plt.colorbar(im6, ax=axes[1, 2], label='(Pred - GT) / GT')

    plt.tight_layout()
    plt.savefig('fwi_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.show()










# Create figure with subplots for better comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Ground Truth
    im1 = axes[0, 0].imshow(gt, cmap='viridis', vmin=1400, vmax=1700)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Z')
    plt.colorbar(im1, ax=axes[0, 0], label='Vp (m/s)')

    # Plot 2: Mask
    im2 = axes[0, 1].imshow(mask3, cmap='gray')
    axes[0, 1].set_title('Binary Mask')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes[0, 1], label='Mask Value')

    # Plot 3: NN Prediction
    im3 = axes[0, 2].imshow(vp_NN, cmap='viridis', vmin=1400, vmax=1700)
    axes[0, 2].set_title('FWI with NN Starting Model')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Z')
    plt.colorbar(im3, ax=axes[0, 2], label='Vp (m/s)')

    # Plot 4: Homogeneous Starting Model Result
    im4 = axes[1, 0].imshow(vp_homo, cmap='viridis', vmin=1400, vmax=1700)
    axes[1, 0].set_title('FWI with Homogeneous Starting Model')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Z')
    plt.colorbar(im4, ax=axes[1, 0], label='Vp (m/s)')

    # Calculate differences
    diff_homo = vp_homo - gt
    diff_NN = vp_NN - gt

    # Normalize to GT
    norm_diff_homo = diff_homo / gt
    norm_diff_NN = diff_NN / gt

    # Plot 5: Normalized Difference (Homo)
    im5 = axes[1, 1].imshow(norm_diff_homo, cmap='RdBu_r', vmax=0.05, vmin=-0.05)
    axes[1, 1].set_title('Normalized Error: Homogeneous')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Z')
    plt.colorbar(im5, ax=axes[1, 1], label='(Pred - GT) / GT')

    # Plot 6: Normalized Difference (NN)
    im6 = axes[1, 2].imshow(norm_diff_NN, cmap='RdBu_r', vmax=0.05, vmin=-0.05)
    axes[1, 2].set_title('Normalized Error: NN')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Z')
    plt.colorbar(im6, ax=axes[1, 2], label='(Pred - GT) / GT')

    plt.tight_layout()
    plt.savefig('fwi_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.show()













    # NEW FIGURE: Masked differences
    fig1_5, axes1_5 = plt.subplots(2, 2, figsize=(14, 10))

    # Calculate raw differences (not normalized)
    diff_homo_gt = (vp_homo - gt) * mask
    diff_NN_gt = (vp_NN - gt) * mask
    diff_homo_NN = (vp_homo - vp_NN) * mask

    # Plot 1: Masked difference Homo - GT
    im1 = axes1_5[0, 0].imshow(diff_homo_gt, cmap='RdBu_r', vmin=-50, vmax=50)
    axes1_5[0, 0].set_title('Masked Difference: Homo - GT')
    axes1_5[0, 0].set_xlabel('X')
    axes1_5[0, 0].set_ylabel('Z')
    plt.colorbar(im1, ax=axes1_5[0, 0], label='Vp difference (m/s)')

    # Plot 2: Masked difference NN - GT
    im2 = axes1_5[0, 1].imshow(diff_NN_gt, cmap='RdBu_r', vmin=-50, vmax=50)
    axes1_5[0, 1].set_title('Masked Difference: NN - GT')
    axes1_5[0, 1].set_xlabel('X')
    axes1_5[0, 1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes1_5[0, 1], label='Vp difference (m/s)')

    # Plot 3: Masked difference Homo - NN
    im3 = axes1_5[1, 0].imshow(diff_homo_NN, cmap='RdBu_r', vmin=-50, vmax=50)
    axes1_5[1, 0].set_title('Masked Difference: Homo - NN')
    axes1_5[1, 0].set_xlabel('X')
    axes1_5[1, 0].set_ylabel('Z')
    plt.colorbar(im3, ax=axes1_5[1, 0], label='Vp difference (m/s)')

    # Plot 4: |Masked difference Homo - NN|
    im4 = axes1_5[1, 1].imshow(np.abs(diff_homo_NN), cmap='hot', vmin=0, vmax=50)
    axes1_5[1, 1].set_title('|Masked Difference: Homo - NN|')
    axes1_5[1, 1].set_xlabel('X')
    axes1_5[1, 1].set_ylabel('Z')
    plt.colorbar(im4, ax=axes1_5[1, 1], label='|Vp difference| (m/s)')

    plt.tight_layout()
    plt.savefig('fwi_masked_differences.png', dpi=300, bbox_inches='tight')
    plt.show()











 # NEW FIGURE: Masked differences
    fig1_5, axes1_5 = plt.subplots(2, 2, figsize=(14, 10))

    # Calculate raw differences (not normalized)
    diff_homo_gt = (vp_homo - gt) * mask3
    diff_NN_gt = (vp_NN - gt) * mask3
    diff_homo_NN = (vp_homo - vp_NN) * mask3

    # Plot 1: Masked difference Homo - GT
    im1 = axes1_5[0, 0].imshow(diff_homo_gt, cmap='RdBu_r', vmin=-50, vmax=50)
    axes1_5[0, 0].set_title('Masked Difference: Homo - GT')
    axes1_5[0, 0].set_xlabel('X')
    axes1_5[0, 0].set_ylabel('Z')
    plt.colorbar(im1, ax=axes1_5[0, 0], label='Vp difference (m/s)')

    # Plot 2: Masked difference NN - GT
    im2 = axes1_5[0, 1].imshow(diff_NN_gt, cmap='RdBu_r', vmin=-50, vmax=50)
    axes1_5[0, 1].set_title('Masked Difference: NN - GT')
    axes1_5[0, 1].set_xlabel('X')
    axes1_5[0, 1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes1_5[0, 1], label='Vp difference (m/s)')

    # Plot 3: Masked difference Homo - NN
    im3 = axes1_5[1, 0].imshow(diff_homo_NN, cmap='RdBu_r', vmin=-50, vmax=50)
    axes1_5[1, 0].set_title('Masked Difference: Homo - NN')
    axes1_5[1, 0].set_xlabel('X')
    axes1_5[1, 0].set_ylabel('Z')
    plt.colorbar(im3, ax=axes1_5[1, 0], label='Vp difference (m/s)')

    # Plot 4: |Masked difference Homo - NN|
    im4 = axes1_5[1, 1].imshow(np.abs(diff_homo_NN), cmap='hot', vmin=0, vmax=50)
    axes1_5[1, 1].set_title('|Masked Difference: Homo - NN|')
    axes1_5[1, 1].set_xlabel('X')
    axes1_5[1, 1].set_ylabel('Z')
    plt.colorbar(im4, ax=axes1_5[1, 1], label='|Vp difference| (m/s)')

    plt.tight_layout()
    plt.savefig('fwi_masked_differences.png', dpi=300, bbox_inches='tight')
    plt.show()


















    # Second figure: Comparison with mask applied
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

    # Apply mask
    abs_norm_diff_homo = np.abs(norm_diff_homo)
    abs_norm_diff_NN = np.abs(norm_diff_NN)

    masked_diff_homo = abs_norm_diff_homo * mask
    masked_diff_NN = abs_norm_diff_NN * mask

    # Plot absolute normalized errors
    im1 = axes2[0, 0].imshow(abs_norm_diff_homo, cmap='hot', vmin=0, vmax=0.2)
    axes2[0, 0].set_title('Absolute Normalized Error: Homogeneous')
    axes2[0, 0].set_xlabel('X')
    axes2[0, 0].set_ylabel('Z')
    plt.colorbar(im1, ax=axes2[0, 0], label='|Error|')

    im2 = axes2[0, 1].imshow(abs_norm_diff_NN, cmap='hot', vmin=0, vmax=0.2)
    axes2[0, 1].set_title('Absolute Normalized Error: NN')
    axes2[0, 1].set_xlabel('X')
    axes2[0, 1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes2[0, 1], label='|Error|')

    # Plot masked errors
    im3 = axes2[1, 0].imshow(masked_diff_homo, cmap='hot', vmin=0, vmax=0.2)
    axes2[1, 0].set_title('Masked Error: Homogeneous')
    axes2[1, 0].set_xlabel('X')
    axes2[1, 0].set_ylabel('Z')
    plt.colorbar(im3, ax=axes2[1, 0], label='|Error| (masked)')

    im4 = axes2[1, 1].imshow(masked_diff_NN, cmap='hot', vmin=0, vmax=0.2)
    axes2[1, 1].set_title('Masked Error: NN')
    axes2[1, 1].set_xlabel('X')
    axes2[1, 1].set_ylabel('Z')
    plt.colorbar(im4, ax=axes2[1, 1], label='|Error| (masked)')

    plt.tight_layout()
    plt.savefig('fwi_comparison_masked.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Third figure: Direct comparison
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Difference between the two approaches
    diff = abs_norm_diff_homo - abs_norm_diff_NN
    masked_diff = diff * mask

    im1 = axes3[0].imshow(diff, cmap='coolwarm', vmin=-0.025, vmax=0.025)
    axes3[0].set_title('Improvement: |Homo Error| - |NN Error|\n(positive = NN better)')
    axes3[0].set_xlabel('X')
    axes3[0].set_ylabel('Z')
    plt.colorbar(im1, ax=axes3[0], label='Difference')

    im2 = axes3[1].imshow(masked_diff, cmap='coolwarm', vmin=-0.025, vmax=0.025)
    axes3[1].set_title('Masked Improvement\n(positive = NN better)')
    axes3[1].set_xlabel('X')
    axes3[1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes3[1], label='Difference (masked)')

    plt.tight_layout()
    plt.savefig('fwi_comparison_improvement.png', dpi=300, bbox_inches='tight')
    plt.show()












  # Second figure: Comparison with mask applied
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

    # Apply mask
    abs_norm_diff_homo = np.abs(norm_diff_homo)
    abs_norm_diff_NN = np.abs(norm_diff_NN)

    masked_diff_homo = abs_norm_diff_homo * mask3
    masked_diff_NN = abs_norm_diff_NN * mask3

    # Plot absolute normalized errors
    im1 = axes2[0, 0].imshow(abs_norm_diff_homo, cmap='hot', vmin=0, vmax=0.2)
    axes2[0, 0].set_title('Absolute Normalized Error: Homogeneous')
    axes2[0, 0].set_xlabel('X')
    axes2[0, 0].set_ylabel('Z')
    plt.colorbar(im1, ax=axes2[0, 0], label='|Error|')

    im2 = axes2[0, 1].imshow(abs_norm_diff_NN, cmap='hot', vmin=0, vmax=0.2)
    axes2[0, 1].set_title('Absolute Normalized Error: NN')
    axes2[0, 1].set_xlabel('X')
    axes2[0, 1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes2[0, 1], label='|Error|')

    # Plot masked errors
    im3 = axes2[1, 0].imshow(masked_diff_homo, cmap='hot', vmin=0, vmax=0.2)
    axes2[1, 0].set_title('Masked Error: Homogeneous')
    axes2[1, 0].set_xlabel('X')
    axes2[1, 0].set_ylabel('Z')
    plt.colorbar(im3, ax=axes2[1, 0], label='|Error| (masked)')

    im4 = axes2[1, 1].imshow(masked_diff_NN, cmap='hot', vmin=0, vmax=0.2)
    axes2[1, 1].set_title('Masked Error: NN')
    axes2[1, 1].set_xlabel('X')
    axes2[1, 1].set_ylabel('Z')
    plt.colorbar(im4, ax=axes2[1, 1], label='|Error| (masked)')

    plt.tight_layout()
    plt.savefig('fwi_comparison_masked.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Third figure: Direct comparison
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # Difference between the two approaches
    diff = abs_norm_diff_homo - abs_norm_diff_NN
    masked_diff = diff * mask3

    im1 = axes3[0].imshow(diff, cmap='coolwarm', vmin=-0.025, vmax=0.025)
    axes3[0].set_title('Improvement: |Homo Error| - |NN Error|\n(positive = NN better)')
    axes3[0].set_xlabel('X')
    axes3[0].set_ylabel('Z')
    plt.colorbar(im1, ax=axes3[0], label='Difference')

    im2 = axes3[1].imshow(masked_diff, cmap='coolwarm', vmin=-0.025, vmax=0.025)
    axes3[1].set_title('Masked Improvement\n(positive = NN better)')
    axes3[1].set_xlabel('X')
    axes3[1].set_ylabel('Z')
    plt.colorbar(im2, ax=axes3[1], label='Difference (masked)')

    plt.tight_layout()
    plt.savefig('fwi_comparison_improvement.png', dpi=300, bbox_inches='tight')
    plt.show()













    # Print quantitative comparison
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISON")
    print("="*60)
    print(f"\nFull Domain:")
    print(f"  Mean Absolute Error (Homo): {np.mean(abs_norm_diff_homo):.4f}")
    print(f"  Mean Absolute Error (NN):   {np.mean(abs_norm_diff_NN):.4f}")
    print(f"  Improvement:                 {(np.mean(abs_norm_diff_homo) - np.mean(abs_norm_diff_NN))/np.mean(abs_norm_diff_homo)*50:.2f}%")

    print(f"\nMasked Region (ROI):")
    masked_pixels = mask > 0
    if np.any(masked_pixels):
        mae_homo_masked = np.mean(abs_norm_diff_homo[masked_pixels])
        mae_nn_masked = np.mean(abs_norm_diff_NN[masked_pixels])
        print(f"  Mean Absolute Error (Homo): {mae_homo_masked:.4f}")
        print(f"  Mean Absolute Error (NN):   {mae_nn_masked:.4f}")
        print(f"  Improvement:                 {(mae_homo_masked - mae_nn_masked)/mae_homo_masked*50:.2f}%")
        
        # Additional stats for the new figure
        print(f"\n  Mean Difference Homo-GT (in ROI):  {np.mean(diff_homo_gt[masked_pixels]):.2f} m/s")
        print(f"  Mean Difference NN-GT (in ROI):    {np.mean(diff_NN_gt[masked_pixels]):.2f} m/s")
        print(f"  Mean Difference Homo-NN (in ROI):  {np.mean(diff_homo_NN[masked_pixels]):.2f} m/s")
        print(f"  Mean |Difference| Homo-NN (in ROI): {np.mean(np.abs(diff_homo_NN[masked_pixels])):.2f} m/s")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()